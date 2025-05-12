import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import ChatDataset

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