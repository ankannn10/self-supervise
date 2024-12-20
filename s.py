#self-supervise

import os
from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed
)
import torch

# -----------------------------
# Functions
# -----------------------------

def get_device():
    """
    Get the best available device for computation (CUDA, MPS, or CPU).
    Returns:
        str: The device type (e.g., "cuda", "mps", or "cpu").
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_and_preprocess_dataset(csv_path: str, tokenizer):
    """
    Load the dataset from a CSV file using the datasets library.
    The CSV should have columns: 'transcript', 'comment'.

    Args:
        csv_path (str): Path to the CSV file.
        tokenizer: The tokenizer to use for [SEP] token.
    Returns:
        dataset: A Hugging Face dataset with the combined text field.
    """
    dataset = load_dataset("csv", data_files=csv_path)
    # Combine transcript and comment into a single text field using tokenizer.sep_token
    def combine_texts(examples):
        combined = []
        for t, c in zip(examples["transcript"], examples["comment"]):
            combined.append(f"{t} {tokenizer.sep_token} {c}")
        return {"text": combined}

    # Map the function to transform the dataset
    dataset = dataset.map(combine_texts, batched=True)
    return dataset


def tokenize_function(examples, tokenizer, max_length=512):
    """
    Tokenize the text using RobertaTokenizer.

    Args:
        examples (dict): A batch of examples with a 'text' key.
        tokenizer (RobertaTokenizerFast): The tokenizer.
        max_length (int): Maximum sequence length.
    Returns:
        dict: A dictionary with tokenized inputs (input_ids, attention_mask).
    """
    return tokenizer(examples["text"], max_length=max_length, truncation=True, padding="max_length")


def create_data_collator(tokenizer):
    """
    Create a DataCollatorForLanguageModeling for MLM.

    Args:
        tokenizer (RobertaTokenizerFast): The tokenizer.
    Returns:
        DataCollatorForLanguageModeling: A data collator configured for MLM.
    """
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=0.15,
        #exclude_special_tokens=True  # Ensures special tokens are not masked
    )


def configure_model(device):
    """
    Load the roberta-large model for masked language modeling.

    Args:
        device (str): The device type ("cuda", "mps", or "cpu").
    Returns:
        model: A Roberta model for MLM.
    """
    model = AutoModelForMaskedLM.from_pretrained("roberta-base").to(device)
    return model


def train_model(tokenized_dataset, tokenizer, model, device, output_dir="./yt-domain-adapted-roberta"):
    """
    Train the model using the Hugging Face Trainer.

    Args:
        tokenized_dataset: A tokenized dataset split.
        tokenizer: The RobertaTokenizerFast.
        model: The MLM model (roberta-large).
        device: The device type ("cuda", "mps", or "cpu").
        output_dir (str): Directory to save the final model and tokenizer.
    """
    # Set seed for reproducibility
    set_seed(42)
    model.to(device)
    # Create a data collator for MLM
    data_collator = create_data_collator(tokenizer)

    # Ensure the dataset has 'train' and 'test' splits
    if "train" not in tokenized_dataset:
        tokenized_dataset = tokenized_dataset["train"].train_test_split(test_size=0.1)
    else:
        tokenized_dataset = tokenized_dataset["train"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        learning_rate=5e-5,
        #evaluation_strategy="steps",
        #eval_steps=1000,
        logging_steps=500,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="none",
        #eval_strategy="no"
        #device=device  # Explicitly specify device
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset, #["train"],
        #eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # Train the model
    trainer.train()

    # Save the final model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


# -----------------------------
# Main Script
# -----------------------------
if __name__ == "__main__":
    # Path to the CSV dataset
    csv_path = 'transcripts_comments.csv'

    # Get the best available device
    device = get_device()
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")

    # Add "Transcript:" and "Comment:" as special tokens
    #special_tokens = {"additional_special_tokens": ["Transcript:", "Comment:"]}
    #tokenizer.add_special_tokens(special_tokens)

    # Load and preprocess the dataset
    dataset = load_and_preprocess_dataset(csv_path, tokenizer)

    # Tokenize the dataset
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=["transcript", "comment", "text"])
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask"])

    # Initialize model
    model = configure_model(device)
    model.resize_token_embeddings(len(tokenizer))  # Resize embeddings to account for new special tokens

    # Train and save the model
    train_model(tokenized_dataset, tokenizer, model, device, output_dir="./yt-domain-adapted-roberta")

    print("Domain adaptation completed and model saved at './yt-domain-adapted-roberta'.")
