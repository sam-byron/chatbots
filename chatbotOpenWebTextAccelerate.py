from multiprocessing import Pool
import os
import math
import random
import torch
import json
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer, get_scheduler
from torch.optim import AdamW
from datasets import load_dataset
from tqdm import tqdm
from accelerate import Accelerator
from dataset import ChatDataset
from utils import save_checkpoint, load_checkpoint, batch_generator_sequential, tokenize_sample, load_chunk, process_and_save_chunk
from evaluation import evaluate_perplexity, create_test_subset
from itertools import chain
import argparse
import glob
from functools import partial
import time


def prepare_data(config, tokenizer, cache_path):
    print("Preparing data...")
    block_size = config["block_size"]
    batch_size = config["batch_size"]

    # Ensure the cache folder exists
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
        print(f"Created cache folder: {cache_path}")

    # Check if chunk files exist
    chunk_paths = glob.glob(os.path.join(cache_path, "chunk_*.pt"))

    if len(chunk_paths) == 0:
        print("No cached chunks found. Tokenizing and caching the dataset...")
        hf_ds = load_dataset("openwebtext", split="train", streaming=False, trust_remote_code=True)
        print("Loading openwebtext dataset from HuggingFace...")

        num_samples = config["num_samples"]
        chunk_size = config["chunk_size"]

        tokenize_with_tokenizer = partial(tokenize_sample, tokenizer=tokenizer)

        print("Processing dataset in chunks...")
        # chunk_args = [
        #     (sample_chunk, i, cache_path, tokenize_with_tokenizer)
        #     for i, sample_chunk in enumerate(batch_generator_sequential(hf_ds, chunk_size, num_samples))
        # ]
        chunk_args = [
            (sample_chunk, i, cache_path, tokenize_with_tokenizer)
            for i, sample_chunk in enumerate(batch_generator_sequential(hf_ds, chunk_size, num_samples, batch_size))
        ]

        with Pool(64) as pool:
            chunk_paths = pool.map(process_and_save_chunk, chunk_args)

        print(f"Processed and saved {len(chunk_paths)} chunks.")

    chunk_paths = glob.glob(os.path.join(cache_path, "chunk_*.pt"))
    print(f"Found {len(chunk_paths)} cached chunks.")

    with Pool(min(96, len(chunk_paths))) as pool:
        tokenized_texts_chunks = pool.map(load_chunk, chunk_paths)
    print(f"Loaded {len(tokenized_texts_chunks)} chunks.")
    tokenized_texts = list(chain.from_iterable(tokenized_texts_chunks))
    tokenized_texts = list(map(torch.tensor, tokenized_texts))
    print(f"Loaded {len(tokenized_texts)} samples from cache.")

    print("Shuffling tokenized texts...")
    random.shuffle(tokenized_texts)
    split_index = int(0.75 * len(tokenized_texts))
    train_texts = tokenized_texts[:split_index]
    test_texts = tokenized_texts[split_index:]

    print(f"Train texts: {len(train_texts)}, Test texts: {len(test_texts)}")

    train_dataset = ChatDataset(train_texts, block_size=block_size)
    test_dataset = ChatDataset(test_texts, block_size=block_size)

    def collate_fn(batch):
        pad_id = tokenizer.pad_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(seq, dtype=torch.long) for seq in batch],
            batch_first=True,
            padding_value=pad_id,
        )
        attention_mask = (input_ids != pad_id).long()
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, test_loader, test_texts


def build_model(config):
    model_config = GPT2Config(
        vocab_size=config["vocab_size"],
        n_positions=config["n_positions"],
        n_embd=config["n_embed"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
    )
    model = GPT2LMHeadModel(model_config)
    return model


def train_loop(accelerator, model, train_loader, test_loader, optimizer, scheduler, config, checkpoint_path, tokenizer):
    num_epochs = config["num_epochs"]
    scaler = torch.cuda.amp.GradScaler()

    checkpoint = load_checkpoint(checkpoint_path)
    start_epoch = 0
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    if start_epoch >= num_epochs:
        print("Training already completed. Exiting.")
        return
    
    # Track time for periodic checkpoint saving
    last_checkpoint_time = time.time()

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            with accelerator.autocast():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                total_loss += loss.item()

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Save checkpoint every 5 minutes
            current_time = time.time()
            if current_time - last_checkpoint_time >= 3 * 60:  # 15 minutes in seconds
                save_checkpoint(epoch, model, optimizer, scheduler, scaler, checkpoint_path)
                last_checkpoint_time = current_time
                print(f"Epoch {epoch + 1}, Step {step + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
                # Evaluate perplexity on a subset of the test set
                # test_subset_loader = create_test_subset(test_texts, 10000, block_size, batch_size, num_cpu, collate_fn)
                # test_subset_loss, perplexity = evaluate_perplexity(model, test_subset_loader, device)
                # print(f"Epoch {epoch + 1} Subset test Loss: {test_subset_loss:.4f}, Perplexity: {perplexity:.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        if accelerator.is_main_process:
            save_checkpoint(epoch, model, optimizer, scheduler, scaler, checkpoint_path)


def main():
    accelerator = Accelerator()
    device = accelerator.device

    parser = argparse.ArgumentParser(description="Chatbot Training Script with Accelerate")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    with open(args.config_path, "r") as config_file:
        config = json.load(config_file)

    checkpoint_path = os.path.join(config["cache_path"], "checkpoint.pt")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    train_loader, test_loader, test_texts = prepare_data(config, tokenizer, config["cache_path"])
    model = build_model(config)

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=config["num_epochs"] * len(train_loader),
    )

    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )

    train_loop(accelerator, model, train_loader, test_loader, optimizer, scheduler, config, checkpoint_path, tokenizer)


if __name__ == "__main__":
    main()