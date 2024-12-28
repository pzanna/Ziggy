# Description: Create a custom tokenizer using the Hugging Face tokenizers library
# Author: Paul Zanna
# Date: 27/12/2024

import os
import argparse
import json
import pandas as pd
import tempfile
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.decoders import ByteLevel, WordPiece
from tokenizers.models import BPE, WordLevel
from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer, WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, AutoTokenizer

def main(word_file, config_path):
    # Initialize a tokenizer
    tokenizer = Tokenizer(WordLevel())
    # tokenizer.pre_tokenizer = ByteLevelPreTokenizer()
    tokenizer.pre_tokenizer = Whitespace()
    # tokenizer.decoder = ByteLevel()
    tokenizer.decoder = WordPiece()
    trainer = WordLevelTrainer(special_tokens=["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"], show_progress=True)

    # Load training data
    print("Loading word file...")

    # Train the tokenizer
    print("Training the tokenizer...")
    tokenizer.train([word_file], trainer)

    print("\nTokenizer contains {} tokens.".format(tokenizer.get_vocab_size()))

    # Save the tokenizer
    print("Saving the tokenizer...")
    tokenizer.save(config_path + "custom_tokenizer.json")

    # Load the custom tokenizer json file as a json object
    with open(config_path + "custom_tokenizer.json", "r") as f:
        tokenizer_config = json.load(f)

    # Set the unknown token
    tokenizer_config["model"]["unk_token"] = "[UNK]"

    # Save the updated configuration
    with open(config_path + "custom_tokenizer.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    print("Updated tokenizer.json with unk_token.")

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

    # Save the updated tokenizer.json
    with open(config_path + "tokenizer.json", "w") as f:
        json.dump(tokenizer_data, f, indent=2)

    print("Saving the vocabulary files...")
    tokenizer.model.save(config_path, "ziggy")
    # rename the vocab file
    os.rename(config_path + "ziggy-vocab.json", config_path + "vocab.json")
   
    # Test the tokenizer
    print("Testing the tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config_path, use_fast=True)
    test_text = "the service provider must ensure that all data is encrypted at rest."
    encoded = tokenizer.encode(test_text)
    print(test_text)
    print(encoded)
    decoded = tokenizer.decode(encoded)
    print(decoded)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Ziggy for multi-label text classification and export it to ONNX format")
    parser.add_argument('--word_file', type=str, required=True, help="Path to the data file containing your training words")
    parser.add_argument('--config_path', type=str, required=True, help="Path to save the tokenizer configuration file")
    args = parser.parse_args()
    main(args.word_file, args.config_path)
