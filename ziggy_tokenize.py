# Description: Create a custom tokenizer using the Hugging Face tokenizers library
# Author: Paul Zanna
# Date: 27/12/2024

import os
import argparse
import json
import pandas as pd
import tempfile
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, AutoTokenizer

def main(word_file, config_path):
    # Initialize a tokenizer
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"])

    # Load training data
    print("Loading training data...")
    clause_data = pd.read_csv(word_file)
    clauses = clause_data['clause'].tolist()

    # Write the combined text to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file:
        temp_file.write(' '.join(clauses))
        temp_file_path = temp_file.name

    # Train the tokenizer
    print("Training the tokenizer...")
    tokenizer.train([temp_file_path], trainer)

    # Save the tokenizer
    print("Saving the tokenizer...")
    tokenizer.save(config_path + "custom_tokenizer.json")

    # Load the custom tokenizer
    custom_tokenizer = PreTrainedTokenizerFast(tokenizer_file=config_path + "custom_tokenizer.json")
    custom_tokenizer.add_special_tokens({
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "unk_token": "[UNK]",
        "mask_token": "[MASK]"
    })

    custom_tokenizer.model_max_length = 512
    # Save the tokenizer for future use
    print("Saving the tokenizer configuration...")
    custom_tokenizer.save_pretrained(config_path)


    # Load the tokenizer.json
    with open(config_path + "tokenizer.json", "r") as f:
        tokenizer_data = json.load(f)

    # Convert merges from arrays to space-separated strings
    if "model" in tokenizer_data and "merges" in tokenizer_data["model"]:
        tokenizer_data["model"]["merges"] = [
            " ".join(merge) for merge in tokenizer_data["model"]["merges"]
        ]

    # Save the updated tokenizer.json
    with open(config_path + "tokenizer.json", "w") as f:
        json.dump(tokenizer_data, f, indent=2)

    print("Merges have been successfully converted!")

    print("Saving the vocabulary files...")
    tokenizer.model.save(config_path, "ziggy")
    # rename the vocab file
    os.rename(config_path + "ziggy-vocab.json", config_path + "vocab.json")
    os.rename(config_path + "ziggy-merges.txt", config_path + "merges.txt")


    # Test the tokenizer
    print("Testing the tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config_path, use_fast=True)
    encoded = tokenizer.encode("This is a test sentence.")
    print(encoded)
    decoded = tokenizer.decode(encoded)
    print(decoded)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Ziggy for multi-label text classification and export it to ONNX format")
    parser.add_argument('--word_file', type=str, required=True, help="Path to the data file containing your training words")
    parser.add_argument('--config_path', type=str, required=True, help="Path to save the tokenizer configuration file")
    args = parser.parse_args()
    main(args.word_file, args.config_path)
