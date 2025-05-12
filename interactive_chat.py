import torch
import json
import argparse
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
import os

def load_checkpoint(checkpoint_path="checkpoint.pt"):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint
    else:
        print("No checkpoint found. Starting from scratch.")
        return None

def start_chat_session(model_path, config):
    """Start an interactive chat session with the model."""
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Load the tokenizer using the model name from the config
    tokenizer = GPT2Tokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    # Create the model configuration using the custom parameters
    model_config = GPT2Config(
        vocab_size=config["vocab_size"],
        n_positions=config["n_positions"],
        n_embd=config["n_embed"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
    )

    # Initialize the model with the custom configuration
    model = GPT2LMHeadModel(model_config)

    # Load the model weights from the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    print("Chat session started (type 'quit' to exit)")

    # Initialize conversation history
    conversation_history = ""

    while True:
        text = input("You: ")
        if text.lower() == "quit":
            break

        # Append user input to conversation history
        conversation_history = f"{text}\n"

        # Define max_new_tokens as a variable for consistency
        max_new_tokens = config.get("max_new_tokens", 50)  # Default to 50 if not specified

        # Truncate conversation history to fit within the model's maximum sequence length
        max_history_length = tokenizer.model_max_length - max_new_tokens - 10  # Reserve space for the bot's response
        tokenized_history = tokenizer(conversation_history, truncation=True, max_length=max_history_length, return_tensors="pt")
        conversation_history = tokenizer.decode(tokenized_history["input_ids"][0], skip_special_tokens=True)

        # Tokenize the conversation history
        encoded = tokenizer(
            conversation_history,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Generate the bot's response
        output_ids = model.module.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=config["temperature"],
            top_p=config["top_p"],
            pad_token_id=tokenizer.eos_token_id,  # Silences the warning
        )

        # Decode the response
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        # Append the bot's response to the conversation history
        conversation_history += f"Bot: {response}\n"

        # Print the bot's response
        print(f"Bot: {response}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Start an interactive chat session with a GPT-2 model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration JSON file.")
    args = parser.parse_args()

    # Load the configuration file
    with open(args.config_path, "r") as config_file:
        config = json.load(config_file)

    # Start the chat session
    start_chat_session(model_path=args.model_path, config=config)