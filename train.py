import os
import torch
from math import exp

import click
from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from model import *
import wandb
import json
import logging

# =========================
# Logging Configuration
# =========================

# Create a logger for the training script
logger = logging.getLogger('train_logger')
logger.setLevel(logging.INFO)  # Set to INFO or DEBUG based on your needs

# Create a file handler to write logs to 'train_log.txt'
log_file = os.path.join(os.path.dirname(__file__), "train_log.txt")
file_handler = logging.FileHandler(log_file, mode='w')  # 'w' to overwrite each run
file_handler.setLevel(logging.INFO)  # Adjust level as needed

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)

# Optionally, add a stream handler to also output to console (set to WARNING to reduce verbosity)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.WARNING)  # Only WARNING and above will be printed to console
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# =========================
# Constants and Paths
# =========================

_ROOT = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = os.path.join(_ROOT, "log")
MODEL_DIR = os.path.join(_ROOT, "model")

BPE_MODEL_PATH = os.path.join(MODEL_DIR, "tokenizer.model")
PROCESS_DATA_TXT = os.path.join(_ROOT, "data", "processed.txt")

# Initialize the tokenizer
tokenizer = SentencePieceProcessor(model_file=BPE_MODEL_PATH)
logger.info(f"Tokenizer pad ID: {tokenizer.pad_id()}")

# =========================
# Dataset and Dataloader
# =========================

def preprocess(doc, max_length=512):
    logger.debug(f"Preprocessing document: {doc.strip()}")
    inputs = tokenizer.encode_as_ids("<s>" + doc)
    targets = tokenizer.encode_as_ids(doc + "</s>")

    logger.debug(f"Inputs: {inputs}")
    logger.debug(f"Targets: {targets}")

    return inputs[:max_length], targets[:max_length]


class CustomDataset(Dataset):
    def __init__(self, data_file, sp_model):
        self.data_x = []
        self.data_y = []
        self.sp_model = sp_model

        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Tokenize the sentence and convert to IDs
                    tokens = [3] + self.sp_model.EncodeAsIds(line) + [4]
                    if len(tokens) < 64:
                        continue
                    # For language modeling task, x and y are the same sequence, shifted by one token
                    self.data_x.append(tokens[:-1])  # Input sequence (all tokens except the last one)
                    self.data_y.append(tokens[1:])  # Target sequence (all tokens except the first one)

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return torch.tensor(self.data_x[idx], dtype=torch.long), torch.tensor(self.data_y[idx], dtype=torch.long)


def create_dataloader(dataset, batch_size):
    def collate_fn(batch):
        # Transpose batch of tuples
        batch_x, batch_y = zip(*batch)
        # Pad sequences in each batch
        batch_x = pad_sequence(batch_x, batch_first=True, padding_value=tokenizer.pad_id())
        batch_y = pad_sequence(batch_y, batch_first=True, padding_value=tokenizer.pad_id())

        return batch_x, batch_y

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader

# =========================
# Training Function
# =========================

@click.command()
@click.option('--num-layers', type=int, default=6, show_default=True, help="No. of decoder layers")
@click.option('--hidden-size', type=int, default=512, show_default=True, help="Hidden size")
@click.option('--num-heads', type=int, default=8, show_default=True, help="Number of heads")
@click.option('--max-seq-len', type=int, default=512, show_default=True, help="Sequence length")
@click.option('--vocab-size', type=int, default=32000, show_default=True, help="Vocabulary size")
@click.option('--batch-size', type=int, default=8, show_default=True, help="Batch size")
@click.option('--learning-rate', type=float, default=1e-4, show_default=True, help="Learning rate")
@click.option('--epoch', type=int, default=10, show_default=True, help="Number of epochs")
def train(num_layers, hidden_size, num_heads, max_seq_len, vocab_size, batch_size, learning_rate, epoch):
    logger.info("Starting training process...")

    # Load tokenizer
    tokenizer = SentencePieceProcessor(model_file=BPE_MODEL_PATH)
    logger.info("Tokenizer loaded successfully.")

    # Initialize dataset and dataloader
    dataset = CustomDataset(PROCESS_DATA_TXT, tokenizer)
    dataloader = create_dataloader(dataset, batch_size)
    logger.info(f"Dataset and dataloader created with batch size {batch_size}.")

    # Configuration for the model
    config = {
        "vocab_size": vocab_size,
        "n_head": num_heads,
        "hidden_size": hidden_size,
        "n_layer": num_layers,
        "n_embd": hidden_size,
        "n_local_heads": 23,  # Ensure this parameter is necessary; adjust if needed
        "n_local_kv_heads": 2,  # Updated to 2 to divide num_heads=8
        "eps": 1e-6,
        "max_len": max_seq_len,
        "rope_theta": 1.0,
        "num_key_value_heads": 2,  # Updated to 2 to divide num_heads=8
        "attention_dropout": 0.25,
        "rms_norm_eps": 1.0,
        "weight_decay": 0.1,
        "block_size": max_seq_len
    }

    # Hyperparameter Validation
    if num_heads % config["num_key_value_heads"] != 0:
        logger.error("num_heads must be divisible by num_key_value_heads.")
        raise ValueError("num_heads must be divisible by num_key_value_heads.")

    # Initialize the model
    model = LLAMA(config)
    model._init_weights(model)
    logger.info("Model initialized and weights are set.")

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Model moved to device: {device}")

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=config["weight_decay"]
    )
    logger.info(f"Optimizer initialized with learning rate {learning_rate} and weight decay {config['weight_decay']}.")

    # Initialize WandB
    wandb.init(
        project="miniLLAMA",
        config=config,
        name="run-" + os.path.basename(os.path.dirname(__file__))
    )
    logger.info("WandB initialized.")

    counter = 0  # Global step counter

    # Training loop
    for epoch_idx in range(epoch):
        logger.info(f"Starting epoch {epoch_idx + 1}/{epoch}")

        model.train()
        for batch_x, batch_y in dataloader:
            logger.debug(f"batch_x shape: {batch_x.shape}, batch_y shape: {batch_y.shape}")

            # Skip empty or zero-length batches
            if batch_x.size(1) == 0 or batch_y.size(1) == 0:
                logger.warning("Skipping empty or zero-length batch.")
                continue

            optimizer.zero_grad()
            logits, loss = model(batch_x.to(device), batch_y.to(device))

            if loss is None:
                logger.warning("Loss is None. Skipping this batch.")
                continue

            perplexity = exp(loss.item())

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)  # Adjust max_norm as needed
            optimizer.step()

            logger.info(f"Epoch: {epoch_idx + 1}, Step: {counter + 1}, Loss: {loss.item():.4f}, Perplexity: {perplexity:.2f}")
            wandb.log({"Train Loss": loss.item(), "Train Perplexity": perplexity}, step=counter)
            counter += 1

            # Clean up
            del loss
            del logits
            del perplexity

        # Evaluation Phase (Optional: Use a separate validation dataloader if available)
        logger.info("Starting evaluation phase.")
        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_samples = 0
            for inputs, targets in dataloader:
                logits, loss = model(inputs.to(device), targets.to(device))
                if loss is not None:
                    total_loss += loss.item() * inputs.size(0)
                    total_samples += inputs.size(0)

            if total_samples > 0:
                avg_loss = total_loss / total_samples
                avg_perplexity = exp(avg_loss)
                logger.info(f"Epoch {epoch_idx + 1}, Avg. Loss: {avg_loss:.4f}, Avg. Perplexity: {avg_perplexity:.2f}")
                wandb.log({"Val Loss": avg_loss, "Val Perplexity": avg_perplexity}, step=counter)
            else:
                logger.warning("No samples to evaluate.")

    # Save the trained model
    model_save_path = os.path.join(MODEL_DIR, "llama.bin")
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Model saved to {model_save_path}")

    # Save the configuration
    config_save_path = os.path.join(MODEL_DIR, 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config, f)
    logger.info(f"Configuration saved to {config_save_path}")

    logger.info("Training process completed successfully.")

if __name__ == "__main__":
    train()
