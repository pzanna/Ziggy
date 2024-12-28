# ziggy_train.py

## Description

Train Ziggy for multi-label text classification and export it to ONNX format using a custom tokenizer.

## Usage

```
python ziggy_train.py --model_file <name_of_model> --onnx_file <name_of_onnx_model> --quant_file <name_of_quantized_model> --data_file <name_of_data_file> --vocab_path <path_to_vocab_files> --req_file <name_of_requirements_file>
```

## Command line arguments

`--model_file` Name and path to save the model file.

`--onnx_file` Name and path to save the ONNX model file.

`--quant_file` Name and path to save the Quantized model file.

`--data_file` Name and path to the data file.

`--vocab_path` Path to the vocab configuration files.

`--req_file` Name and path to the requirements file.

## Steps performed

- Parses command-line arguments for model and data file paths.
- Loads a pre-trained tokenizer and defines a function to tokenize and preprocess text.
- Defines hyperparameters for the model and quantization configurations.
- Defines a custom dataset class for text classification.
- Defines a Transformer-based neural network model for multi-label text classification.
- Implements training and evaluation functions for the model.
- Loads training data and labels, and prepares a DataLoader for training.
- Initializes and trains the model, saving the trained model to a specified file.
- Exports the trained model to ONNX format and performs model slimming and quantization.
- Saves the quantized model and prints model parameter counts.

## Functions

`encode_text(text, max_length)` Tokenizes and preprocesses text.
`train_model(model, dataloader, epochs, learning_rate, device)` Trains the model.
`evaluate_model(model, dataloader, device)` Evaluates the model.
`count_parameters(model)` Counts the total, trainable, and non-trainable parameters of the model.

## Classes

`TextClassificationDataset` Custom dataset class for text classification.
`TransformerClassifier` Transformer-based neural network model for multi-label text classification.

## Output

Saves the model in PyTorch (.bin), ONNX (.onnx) and `int8` Quantized ONNX (-quantized.onnx) and print model stats.
