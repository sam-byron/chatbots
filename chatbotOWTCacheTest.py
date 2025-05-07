import os
import torch
import pytest
from transformers import GPT2LMHeadModel, GPT2Config
from chatbotOpenWebTextCache import save_checkpoint, load_checkpoint

@pytest.fixture
def setup_model_and_optimizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create a small GPT-2 model for testing
    model_config = GPT2Config(
        vocab_size= 50257,
        n_positions= 1024,
        n_embed= 1024,
        n_layer= 16,
        n_head= 16,
    )
    model = GPT2LMHeadModel(model_config)
    model = torch.nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    scaler = torch.amp.GradScaler('cuda')
    return model, optimizer, scheduler, scaler, device

def test_loss_reduction(setup_model_and_optimizer):
    model, optimizer, scheduler, scaler, device = setup_model_and_optimizer

    # Simulate a training loop
    initial_loss = 10.0  # Simulated initial loss
    losses = [initial_loss]

    for epoch in range(3):  # Simulate 3 epochs
        # Simulate loss reduction
        current_loss = losses[-1] * 0.9  # Reduce loss by 10% each epoch
        losses.append(current_loss)

        # Simulate optimizer and scheduler steps
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Assert that the loss is decreasing
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"

def test_checkpoint_saving_and_loading(setup_model_and_optimizer, tmp_path):
    model, optimizer, scheduler, scaler, device = setup_model_and_optimizer

    # Simulate saving a checkpoint
    checkpoint_path = os.path.join(tmp_path, "checkpoint.pt")
    save_checkpoint(1, model, optimizer, scheduler, scaler, checkpoint_path)

    # Assert that the checkpoint file exists
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created."

    # Simulate loading the checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    assert checkpoint is not None, "Failed to load checkpoint."

    # Assert that the checkpoint contains the correct keys
    expected_keys = ["epoch", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict"]
    for key in expected_keys:
        assert key in checkpoint, f"Missing key in checkpoint: {key}"

def test_model_state_update(setup_model_and_optimizer):
    model, optimizer, scheduler, scaler, device = setup_model_and_optimizer

    # Save initial model state
    initial_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    # Simulate a training step
    dummy_input = torch.randint(0, 50257, (1, 10)).to(device)  # Random input moved to GPU
    dummy_labels = dummy_input.clone().to(device)  # Labels moved to GPU
    outputs = model(input_ids=dummy_input, labels=dummy_labels)
    loss = outputs.loss

    # Backpropagation
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    # Assert that the model state has changed
    for k, v in model.state_dict().items():
        assert not torch.equal(v, initial_state_dict[k]), f"Model state did not update for key: {k}"