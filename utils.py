import os
import random
import torch
from itertools import islice
from multiprocessing import Pool

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
    """Generate batches of samples from a streaming dataset more efficiently."""
    iterator = iter(dataset)  # Create an iterator from the dataset
    num_samples = 0

    while num_samples < max_samples:
        # Use islice to fetch a batch of size `batch_size`
        batch = list(islice(iterator, batch_size))
        if not batch:
            break  # Stop if there are no more samples
        yield batch
        num_samples += len(batch)

def fetch_batch(args):
    dataset, start, end = args
    return list(islice(dataset, start, end))

def batch_generator_parallel(dataset, batch_size, max_samples, num_workers):
    """Generate batches in parallel using multiprocessing."""
    total_batches = (max_samples + batch_size - 1) // batch_size
    args = [(dataset, i * batch_size, (i + 1) * batch_size) for i in range(total_batches)]

    with Pool(num_workers) as pool:
        for batch in pool.imap(fetch_batch, args):
            yield batch

def tokenize_sample(sample, tokenizer):
    """Helper function to tokenize a single sample."""
    return tokenizer.encode(sample["text"] + tokenizer.eos_token, add_special_tokens=False, truncation=True)

def load_chunk(chunk_path):
    """Helper function to load a single chunk."""
    print(f"Loading chunk from {chunk_path}...")
    return torch.load(chunk_path, map_location="cpu")

def process_and_save_chunk(args):
    """Tokenize and save a single chunk."""
    sample_chunk, chunk_index, cache_path, tokenize_with_tokenizer = args
    print(f"Processing chunk {chunk_index + 1}...")

    # Tokenize the chunk (sequentially within this process)
    tokenized_chunk = list(map(tokenize_with_tokenizer, sample_chunk))

    # Save the tokenized chunk
    chunk_path = os.path.join(cache_path, f"chunk_{chunk_index + 1}.pt")
    torch.save(tokenized_chunk, chunk_path)
    print(f"Saved chunk {chunk_index + 1} to {chunk_path}")
    return chunk_path