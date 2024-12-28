# ziggy_tokenize.py

## Description

This script creates a custom tokenizer using the Hugging Face tokenizers library. The tokenizer is trained on a provided word file and saved to a specified configuration path. It also supports special tokens and can be used for multi-label text classification.

## Usage

```
python ziggy_tokenize.py --word_file <path_to_word_file> --config_path <path_to_config_path>
```

## Command line arguments

`--word_file` Path to the data file containing your training words.

`--config_path` Path to save the tokenizer configuration file.

## Steps performed

## Functions

`main(word_file, config_path)` Trains and saves a custom tokenizer based on the provided word file and configuration path.

## Outputs

Saves tokenizer configuration files and displays the number of tokens. It also varifies the tokenizer by encoding, then decoding, a test sentence.
