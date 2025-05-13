import torch
import json
import argparse
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
import os
import torch.nn.functional as F

def load_checkpoint(checkpoint_path="checkpoint.pt"):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint
    else:
        print("No checkpoint found. Starting from scratch.")
        return None
    
def log_probs_from_logits(logits, labels):
     logp = F.log_softmax(logits, dim=-1)
     logp_label = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
     return logp_label

def sequence_logprob(model, labels, input_len=0):
        with torch.no_grad():
            output = model(labels)
            log_probs = log_probs_from_logits(
                output.logits[:, :-1, :], labels[:, 1:])
            seq_log_prob = torch.sum(log_probs[:, input_len:])
        return seq_log_prob.cpu().numpy()

def start_chat_session(model_path, config):
    """Start an interactive chat session with the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


    while True:
        text = input("You: ")
        if text.lower() == "quit":
            break

        usr_input = f"{text}\n"

        # Define max_new_tokens as a variable for consistency
        max_new_tokens = config.get("max_new_tokens", 50)  # Default to 50 if not specified

        # Tokenize the conversation history
        encoded = tokenizer(
            usr_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Generate the bot's response
        output_beam = model.module.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            # temperature=config["temperature"],
            # top_p=config["top_p"],
            num_beams=10,
            pad_token_id=tokenizer.eos_token_id,  # Silences the warning
            no_repeat_ngram_size=3
        )

        # Decode the response
        logp = sequence_logprob(model, output_beam, input_len=len(input_ids[0]))
        response = tokenizer.decode(output_beam[0])
        print(f"\nlog-prob: {logp:.2f}")

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