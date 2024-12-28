# ziggy_test.py

## Description

Test the Ziggy ONNX model with a sample input text. This script loads a quantized ONNX model and a tokenizer, processes a sample input text,
and predicts the label using the model. The script also prints the predicted label and its probability.

## Usage

```
python ziggy_test.py --quant_file <path_to_quantized_model> --req_file <path_to_requirements_file> --vocab_path <path_to_tokenizer_config>
```

## Command line arguments

`--quant_file` Path to the quantized model file.
`--req_file` Path to the requirements file (CSV) containing label mappings.
`--vocab_path` Path to the tokenizer configuration files.

## Steps performed

- Define parameters.
- Defines a sample input text for testing
- Loads custom tokenizer from the provided vocabulary path using the Hugging `Face PreTrainedTokenizerFast.` function.
- Loads the quantized ONNX model using `onnxruntime.InferenceSession` function.
- Converts sample input text to lowercase.
- Truncates if the token count exceeds max_seq_length.
- Pads with zeros if the token count is less than max_seq_length.
- Reads label information from a CSV file (req_file).
- Creates label to ID (label2id) and ID to label (id2label) mappings.
- Tokenizes input text.
- Prints the input names, data types, and shapes of the ONNX model inputs for verification.
- Prints the predicted label name and its ID, probability of the predicted label. and all class probabilities.

## Functions

`encode_text(text, max_length)` Tokenizes and preprocesses the input text.
`softmax(logits)` Applies softmax to logits to compute probabilities.

## Outputs

Outputs model metadata, predicted label of the sample text and probalibities array of all labels.
