import os
import math
import torch
import json
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, get_scheduler, AutoTokenizer
from torch.optim import AdamW
from datasets import load_dataset
from tqdm import tqdm
from multiprocessing import Pool
import argparse

# --- Parse Configuration File ---
parser = argparse.ArgumentParser(description="Chatbot Training Script")
parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file")
args = parser.parse_args()

# Load configuration from the specified file
with open(args.config_path, "r") as config_file:
    config = json.load(config_file)

# 1) Hyperparameters & device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_cpu = os.cpu_count()
block_size = config["block_size"]
batch_size = config["batch_size"]
grad_accum = config["grad_accum"]
num_epochs = config["num_epochs"]

# 2) Tokenizer & Model setup
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
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

# 3) Load raw texts and count total chunks
hf_ds = load_dataset(config["dataset_name"], split=config["dataset_split"], num_proc=num_cpu)
print(f"Loaded {len(hf_ds)} samples")
texts = hf_ds["text"]

def count_chunks(text):
    toks = tokenizer.encode(
        text + tokenizer.eos_token,
        add_special_tokens=False,
        truncation=True,
    )
    return len(toks) // block_size

# Use multiprocessing to count chunks in parallel
with Pool(num_cpu) as pool:
    total_chunks = sum(pool.map(count_chunks, texts))

steps_per_epoch = math.ceil(total_chunks / (batch_size * grad_accum))
num_training_steps = steps_per_epoch * num_epochs

print(f"Total chunks: {total_chunks}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Total training steps: {num_training_steps}")

# 4) IterableDataset for on-the-fly tokenization + chunking
class ChatDataset(IterableDataset):
    def __init__(self, texts, tokenizer, block_size=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.total_chunks = self._calculate_total_chunks()

    def _calculate_total_chunks(self):
        total_chunks = 0
        for chat in self.texts:
            ids = self.tokenizer.encode(
                chat + self.tokenizer.eos_token,
                add_special_tokens=False,
                truncation=True,
            )
            total_chunks += len(ids) // self.block_size
        return total_chunks

    def __len__(self):
        return self.total_chunks

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            start, end = 0, len(self.texts)
        else:
            per_worker = math.ceil(len(self.texts) / worker_info.num_workers)
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(self.texts))

        for chat in self.texts[start:end]:
            ids = self.tokenizer.encode(
                chat + self.tokenizer.eos_token,
                add_special_tokens=False,
                truncation=True,
            )
            for i in range(0, len(ids), self.block_size):
                chunk = ids[i : i + self.block_size]
                if len(chunk) == self.block_size:
                    yield chunk

dataset = ChatDataset(texts, tokenizer, block_size=block_size)

# 5) collate_fn: pad to max in-batch length, build attention_mask & labels
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

# 6) DataLoader parallelized over all CPU cores
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,  # IterableDataset + shuffle=True is tricky
    num_workers=num_cpu,
    pin_memory=True,
    collate_fn=collate_fn,
)

# Optimizer, Scheduler, AMP scaler
optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=config["warmup_steps"], num_training_steps=num_training_steps)
scaler = torch.amp.GradScaler('cuda')

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

# --- Load Checkpoint if Available ---
checkpoint_path = config["checkpoint_path"]
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
    gradient_accumulation_steps = grad_accum
    model.train()

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss.mean() / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item() * gradient_accumulation_steps

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        # Save checkpoint after each epoch
        save_checkpoint(epoch, model, optimizer, scheduler, scaler, checkpoint_path)

# --- Interactive Chat Session with History ---
model.eval()
print("Chat session started (type 'quit' to exit)")

# Initialize conversation history
conversation_history = ""

while True:
    text = input("You: ")
    if text.lower() == "quit":
        break

    # Append user input to conversation history
    conversation_history += f"User: {text}\n"

    # Truncate conversation history to fit within the model's maximum sequence length
    max_history_length = tokenizer.model_max_length - config["max_new_tokens"] - 10  # Reserve space for the bot's response
    conversation_history = conversation_history[-max_history_length:]

    # Tokenize the conversation history
    encoded = tokenizer(
        conversation_history + "Bot: ",
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Generate a response using `max_new_tokens`
    output_ids = model.module.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=config["max_new_tokens"],  # Limit the number of tokens generated
        do_sample=True,
        temperature=config["temperature"],
        top_p=config["top_p"],
        pad_token_id=tokenizer.eos_token_id,  # Silences the warning
    )

    # Decode the response and append it to the conversation history
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    conversation_history += f"Bot: {response}\n"

    # Print the bot's response
    print(f"Bot: {response}")