import os
import math
import random
import torch
import json
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, get_scheduler
from torch.optim import AdamW
from datasets import load_dataset
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import argparse
import glob
from functools import partial
from interactive_chat import start_chat_session
import time  # Add this import at the top of the file

class ChatDataset(IterableDataset):
        def __init__(self, tokenized_texts, block_size=256):
            self.tokenized_texts = tokenized_texts
            self.block_size = block_size
            self.total_chunks = self._calculate_total_chunks()

        def _calculate_total_chunks(self):
            total_chunks = 0
            for ids in self.tokenized_texts:
                total_chunks += len(ids) // self.block_size
            return total_chunks

        def __len__(self):
            return self.total_chunks

        def __iter__(self):
            worker_info = get_worker_info()
            if worker_info is None:
                start, end = 0, len(self.tokenized_texts)
            else:
                per_worker = math.ceil(len(self.tokenized_texts) / worker_info.num_workers)
                start = worker_info.id * per_worker
                end = min(start + per_worker, len(self.tokenized_texts))

            for ids in self.tokenized_texts[start:end]:
                for i in range(0, len(ids), self.block_size):
                    chunk = ids[i : i + self.block_size]
                    if len(chunk) == self.block_size:
                        yield chunk

# --- Checkpoint Save/Load Functions ---
def save_checkpoint(epoch, model, optimizer, scheduler, scaler, checkpoint_path="checkpoint.pt"):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch + 1}")

def load_checkpoint(checkpoint_path="checkpoint.pt"):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint
    else:
        print("No checkpoint found. Starting from scratch.")
        return None

def batch_generator(dataset, batch_size, max_samples):
    """Generate batches of samples from a streaming dataset."""
    batch = []
    for i, sample in enumerate(dataset):
        if i >= max_samples:
            break
        batch.append(sample)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:  # Yield the last batch if it's not empty
        yield batch

def tokenize_sample(sample, tokenizer):
    """Helper function to tokenize a single sample."""
    return tokenizer.encode(sample["text"] + tokenizer.eos_token, add_special_tokens=False, truncation=True)

def load_chunk(chunk_path):
    """Helper function to load a single chunk."""
    print(f"Loading chunk from {chunk_path}...")
    return torch.load(chunk_path, map_location="cpu")

def evaluate_perplexity(model, test_loader, device):
    """Evaluate perplexity on the test set."""
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():  # Disable gradient computation
        for batch in tqdm(test_loader, desc="Evaluating Perplexity"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss.mean()

            # Accumulate loss and token count
            total_loss += loss.item() * input_ids.size(0)  # Multiply by batch size
            total_tokens += input_ids.size(0)

    # Calculate average loss and perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return avg_loss, perplexity

def create_test_subset(test_texts, num_samples, block_size, batch_size, num_cpu, collate_fn):
        """Create a subset of the test set and return a DataLoader."""
        random.seed(42)  # Set seed for reproducibility
        subset_indices = random.sample(range(len(test_texts)), num_samples)  # Randomly sample indices
        test_subset_texts = [test_texts[i] for i in subset_indices]  # Create a subset of test_texts
        test_subset = ChatDataset(test_subset_texts, block_size=block_size)  # Create a ChatDataset for the subset
        print(f"Test subset created with {len(test_subset_texts)} samples")

        # Create a DataLoader for the test subset
        test_subset_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_cpu,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        return test_subset_loader

def main():
    # --- Parse Configuration File ---
    parser = argparse.ArgumentParser(description="Chatbot Training Script")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    # Load configuration from the specified file
    with open(args.config_path, "r") as config_file:
        config = json.load(config_file)

    # 1) Hyperparameters & device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_cpu = os.cpu_count() - 4  # Reserve some CPU cores for other tasks
    block_size = config["block_size"]
    batch_size = config["batch_size"]
    grad_accum = config["grad_accum"]
    num_epochs = config["num_epochs"]

    # 2) Tokenizer & Model setup
    tokenizer = GPT2Tokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    model_config = GPT2Config(
        vocab_size=config["vocab_size"],
        n_positions=config["n_positions"],
        n_embd=config["n_embed"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
    )

    # tell the tokenizer what its max context really is
    tokenizer.model_max_length = model_config.n_positions
    tokenizer.init_kwargs["model_max_length"] = model_config.n_positions

    model = GPT2LMHeadModel(model_config)
    model = torch.nn.DataParallel(model)
    model.to(device)

    # Read cache_path and checkpoint_path from the configuration file
    cache_path = args.config_path.replace(".json", "")  # Create a cache folder based on the config file name
    checkpoint_path = os.path.join(cache_path, "checkpoint.pt")  # Save checkpoint under the cache folder

    # Ensure the cache folder exists
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
        print(f"Created cache folder: {cache_path}")

    # Check if chunk files exist
    chunk_paths = glob.glob(os.path.join(cache_path, "chunk_*.pt"))  # Find all chunk files

    if len(chunk_paths) == 0:
        print("No cached chunks found. Tokenizing and caching the dataset...")
        # Load OpenWebText dataset in streaming mode
        hf_ds = load_dataset("openwebtext", split="train", streaming=True, trust_remote_code=True)
        print("Streaming the dataset...")

        # Limit the dataset to 1,000,000 samples and process in chunks
        num_samples = config["num_samples"]
        chunk_size = config["chunk_size"]
        num_workers = min(cpu_count(), 64)  # Use up to 32 CPU cores
        print(f"Using {num_workers} CPU cores for tokenization.")

        # Create a partial function with the tokenizer pre-filled
        tokenize_with_tokenizer = partial(tokenize_sample, tokenizer=tokenizer)

        with Pool(num_workers) as pool:
            for i, sample_chunk in enumerate(batch_generator(hf_ds, chunk_size, num_samples)):
                print(f"Processing chunk {i + 1}...")
                tokenized_chunk = pool.map(tokenize_with_tokenizer, sample_chunk)

                # Save each chunk separately under the cache folder
                chunk_path = os.path.join(cache_path, f"chunk_{i + 1}.pt")
                torch.save(tokenized_chunk, chunk_path)
                print(f"Saved chunk {i + 1} to {chunk_path}")

        # After saving the chunks, load them back into `tokenized_texts`
        print("Loading tokenized dataset from saved chunks...")
        chunk_paths = glob.glob(os.path.join(cache_path, "chunk_*.pt"))  # Find all chunk files
        print(f"Found {len(chunk_paths)} chunks.")

        tokenized_texts = []
        # Use multiprocessing to load chunks in parallel
        with Pool(min(cpu_count(), len(chunk_paths))) as pool:
            tokenized_texts_chunks = pool.map(load_chunk, chunk_paths)

        print(f"Loaded {len(tokenized_texts)} samples from saved chunks.")
    else:
        print(f"Cached chunks found. Loading tokenized dataset from cache...")
        print(f"Found {len(chunk_paths)} chunks.")

        # Use multiprocessing to load chunks in parallel
        with Pool(min(cpu_count(), len(chunk_paths))) as pool:
            tokenized_texts_chunks = pool.map(load_chunk, chunk_paths)

        # Flatten the list of chunks into a single list
        tokenized_texts = [item for sublist in tokenized_texts_chunks for item in sublist]
        print(f"Loaded {len(tokenized_texts)} samples from cache.")

    # 3) Dataset and DataLoader setup
    
    # shuffle tokenized_texts
    print("Shuffling tokenized texts...")
    random.shuffle(tokenized_texts)
    print(f"Shuffled {len(tokenized_texts)} samples.")
    split_index = int(0.75 * len(tokenized_texts))
    train_texts = tokenized_texts[:split_index]
    test_texts = tokenized_texts[split_index:]

    dataset = ChatDataset(train_texts, block_size=block_size)
    test_dataset = ChatDataset(test_texts, block_size=block_size)
    print(f"Dataset created")

    def collate_fn(batch):
        pad_id = tokenizer.pad_token_id
        B = len(batch)
        L = max(len(seq) for seq in batch)

        input_ids = torch.full((B, L), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, L), dtype=torch.long)
        labels = torch.full((B, L), -100, dtype=torch.long)

        for i, seq in enumerate(batch):
            length = len(seq)
            input_ids[i, :length] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, :length] = 1
            labels[i, :length] = torch.tensor(seq, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_cpu,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=2,  # Prefetch 2 batches per worker. No increase in performance with 2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_cpu,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print(f"DataLoader created")
    # Optimizer, Scheduler, AMP scaler
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=config["warmup_steps"], num_training_steps=num_epochs)
    scaler = torch.amp.GradScaler('cuda')

    # --- Load Checkpoint if Available ---
    checkpoint = load_checkpoint(checkpoint_path)

    start_epoch = 0
    if checkpoint:
        model.module.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    if start_epoch >= num_epochs:
        print("Training already completed. Exiting.")
    else:
        print(f"Resuming training from epoch {start_epoch + 1} of {num_epochs}")
        # --- Training Loop with Resumption ---
        model.train()

        # Track time for periodic checkpoint saving
        last_checkpoint_time = time.time()

        for epoch in range(start_epoch, num_epochs):
            total_loss = 0
            print(f"Starting Epoch {epoch + 1}/{num_epochs}...")
            for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss.mean()

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                total_loss += loss.item()

                # Save checkpoint every 15 minutes
                current_time = time.time()
                if current_time - last_checkpoint_time >= 15 * 60:  # 15 minutes in seconds
                    save_checkpoint(epoch, model, optimizer, scheduler, scaler, checkpoint_path)
                    last_checkpoint_time = current_time
                    print(f"Epoch {epoch + 1}, Step {step + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
                    # Evaluate perplexity on a subset of the test set
                    test_subset_loader = create_test_subset(test_texts, 10000, block_size, batch_size, num_cpu, collate_fn)
                    test_subset_loss, perplexity = evaluate_perplexity(model, test_subset_loader, device)
                    print(f"Epoch {epoch + 1} Subset test Loss: {test_subset_loss:.4f}, Perplexity: {perplexity:.4f}")

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

             # Evaluate perplexity on the test set
            test_loss, perplexity = evaluate_perplexity(model, test_loader, device)
            print(f"Epoch {epoch + 1} Test Loss: {test_loss:.4f}, Perplexity: {perplexity:.4f}")

            # Save checkpoint at the end of each epoch
            save_checkpoint(epoch, model, optimizer, scheduler, scaler, checkpoint_path)

    # --- Interactive Chat Session ---
    start_chat_session(checkpoint_path, config=config)

if __name__ == "__main__":
    # torch.cuda.set_per_process_memory_fraction(0.8, device=0)  # Limit to 80% of GPU memory
    main()