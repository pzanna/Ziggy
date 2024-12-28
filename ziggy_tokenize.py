import os
import argparse
import json
from tokenizers import Tokenizer, pre_tokenizers, decoders, models, trainers
from transformers import PreTrainedTokenizerFast

def main(word_file, config_path):
    # Initialize a tokenizer
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece()
    trainer = trainers.WordLevelTrainer(special_tokens=["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"], show_progress=True)

    # Load and train the tokenizer
    print("Loading word file and training the tokenizer...")
    tokenizer.train([word_file], trainer)
    print(f"\nTokenizer contains {tokenizer.get_vocab_size()} tokens.")

    # Save the tokenizer
    tokenizer_path = os.path.join(config_path, "custom_tokenizer.json")
    print("Saving the tokenizer...")
    tokenizer.save(tokenizer_path)

    # Update the tokenizer configuration
    with open(tokenizer_path, "r") as f:
        tokenizer_config = json.load(f)
    tokenizer_config["model"]["unk_token"] = "[UNK]"
    with open(tokenizer_path, "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    print("Updated tokenizer.json with unk_token.")

    # Load and configure the custom tokenizer
    custom_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    custom_tokenizer.add_special_tokens({
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "unk_token": "[UNK]",
        "mask_token": "[MASK]"
    })
    custom_tokenizer.model_max_length = 512

    # Save the tokenizer configuration
    print("Saving the tokenizer configuration...")
    custom_tokenizer.save_pretrained(config_path)

    # Save the vocabulary files
    print("Saving the vocabulary files...")
    tokenizer.model.save(config_path, "ziggy")
    os.rename(os.path.join(config_path, "ziggy-vocab.json"), os.path.join(config_path, "vocab.json"))

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
