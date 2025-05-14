import os
import math
import random
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
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
from dataset import ChatDataset
from utils import (
    save_checkpoint,
    load_checkpoint,
    batch_generator,
    tokenize_sample,
    load_chunk,
    process_and_save_chunk,
    batch_generator_parallel,
    batch_generator_sequential,
)
from evaluation import evaluate_perplexity, create_test_subset
from interactive_chat import start_chat_session

def prepare_data(args, config, tokenizer, num_cpu, cache_path):
    block_size = config["block_size"]
    batch_size = config["batch_size"]
    # Ensure the cache folder exists
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
        if dist.get_rank() == 0:
            print(f"Created cache folder: {cache_path}")

    # Check if chunk files exist
    chunk_paths = glob.glob(os.path.join(cache_path, "chunk_*.pt"))

    if len(chunk_paths) == 0 and dist.get_rank() == 0:
        print("No cached chunks found. Tokenizing and caching the dataset...")
        hf_ds = load_dataset("openwebtext", split="train", streaming=False, trust_remote_code=True)
        print("Streaming the dataset...")

        num_samples = config["num_samples"]
        chunk_size = config["chunk_size"]
        num_workers = min(cpu_count(), 120)
        print(f"Using {num_workers} CPU cores for tokenization.")

        tokenize_with_tokenizer = partial(tokenize_sample, tokenizer=tokenizer)
        print("Processing dataset in chunks...")
        chunk_args = [
            (sample_chunk, i, cache_path, tokenize_with_tokenizer)
            for i, sample_chunk in enumerate(
                batch_generator_parallel(hf_ds, chunk_size, num_samples, num_workers)
            )
        ]
        print(f"Total chunks to process: {len(chunk_args)}")

        with Pool(num_workers) as pool:
            chunk_paths = pool.map(process_and_save_chunk, chunk_args)

        print(f"Processed and saved {len(chunk_paths)} chunks.")

    # Wait for all processes before loading
    dist.barrier()
    chunk_paths = glob.glob(os.path.join(cache_path, "chunk_*.pt"))
    if dist.get_rank() == 0:
        print(f"Cached chunks found. Loading tokenized dataset from cache...")
        print(f"Found {len(chunk_paths)} chunks.")

    with Pool(min(cpu_count(), len(chunk_paths))) as pool:
        tokenized_chunks = pool.map(load_chunk, chunk_paths)

    tokenized_texts = [item for sublist in tokenized_chunks for item in sublist]
    if dist.get_rank() == 0:
        print(f"Loaded {len(tokenized_texts)} samples from cache.")

    # Shuffle and split
    if dist.get_rank() == 0:
        print("Shuffling tokenized texts...")
    random.shuffle(tokenized_texts)
    split_idx = int(0.75 * len(tokenized_texts))
    train_texts = tokenized_texts[:split_idx]
    test_texts = tokenized_texts[split_idx:]

    dataset = ChatDataset(train_texts, block_size=block_size)
    test_dataset = ChatDataset(test_texts, block_size=block_size)
    if dist.get_rank() == 0:
        print("Dataset created")

    def collate_fn(batch):
        pad_id = tokenizer.pad_token_id
        B = len(batch)
        L = max(len(seq) for seq in batch)
        input_ids = torch.full((B, L), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, L), dtype=torch.long)
        labels = torch.full((B, L), -100, dtype=torch.long)
        for i, seq in enumerate(batch):
            l = len(seq)
            input_ids[i, :l] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, :l] = 1
            labels[i, :l] = torch.tensor(seq, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    # Distributed samplers
    train_sampler = DistributedSampler(dataset)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_cpu,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=4,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_cpu,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=4,
        persistent_workers=True,
    )

    if dist.get_rank() == 0:
        print("DataLoader created")
    return train_loader, test_loader, test_texts


def build_model(config, device):
    model_config = GPT2Config(
        vocab_size=config["vocab_size"],
        n_positions=config["n_positions"],
        n_embd=config["n_embed"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        loss_type="cross_entropy",
    )
    model = GPT2LMHeadModel(model_config).to(device)
    model = DDP(model, device_ids=[device.index], output_device=device.index)
    return model


def train_loop(checkpoint_path, config, model, train_loader, test_loader, device, test_texts, tokenizer):
    num_epochs = config["num_epochs"]
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = get_scheduler(
        "cosine", optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=num_epochs * len(train_loader)
    )
    scaler = torch.amp.GradScaler('cuda')

    checkpoint = load_checkpoint(checkpoint_path)
    start_epoch = 0
    if checkpoint:
        model.module.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    if start_epoch >= num_epochs:
        if dist.get_rank() == 0:
            print("Training already completed. Exiting.")
        return

    if dist.get_rank() == 0:
        print(f"Resuming training from epoch {start_epoch+1} of {num_epochs}")
    model.train()
    last_ckpt_time = time.time()

    for epoch in range(start_epoch, num_epochs):
        # set epoch for sampler
        train_loader.sampler.set_epoch(epoch)
        total_loss = 0
        if dist.get_rank() == 0:
            print(f"Starting Epoch {epoch+1}/{num_epochs}...")
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            inputs = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**inputs)
                loss = outputs.loss.mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            total_loss += loss.detach().item()

            # periodic checkpoint
            if time.time() - last_ckpt_time >= 30*60:
                if dist.get_rank() == 0:
                    save_checkpoint(epoch, model, optimizer, scheduler, scaler, checkpoint_path)
                    print(f"Epoch {epoch+1}, Step {step+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
                    test_subset_loader = create_test_subset(
                        test_texts, 10000, config["block_size"], config["batch_size"], os.cpu_count()-4, collate_fn
                    )
                    test_loss, ppl = evaluate_perplexity(model, test_subset_loader, device)
                    print(f"Subset test Loss: {test_loss:.4f}, Perplexity: {ppl:.4f}")
                last_ckpt_time = time.time()

        avg_loss = total_loss / len(train_loader)
        if dist.get_rank() == 0:
            print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
            test_loss, ppl = evaluate_perplexity(model, test_loader, device)
            print(f"Test Loss: {test_loss:.4f}, Perplexity: {ppl:.4f}")
            save_checkpoint(epoch, model, optimizer, scheduler, scaler, checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description="Chatbot Training Script")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)), help="Local process rank")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    num_cpu = os.cpu_count() - 4

    with open(args.config_path, "r") as f:
        config = json.load(f)

    cache_path = args.config_path.replace(".json", "")
    checkpoint_path = os.path.join(cache_path, "checkpoint.pt")

    tokenizer = GPT2Tokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = config["n_positions"]

    train_loader, test_loader, test_texts = prepare_data(
        args, config, tokenizer, num_cpu, cache_path
    )
    model = build_model(config, device)
    train_loop(checkpoint_path, config, model, train_loader, test_loader, device, test_texts, tokenizer)
    if dist.get_rank() == 0:
        start_chat_session(checkpoint_path, config=config)

if __name__ == "__main__":
    main()
