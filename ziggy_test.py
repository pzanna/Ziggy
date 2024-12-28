# Description: Test the Ziggy ONNX model with a sample input text
# Author: Paul Zanna
# Date: 24/12/2024

# Import libraries
import torch
import onnxruntime as ort
from transformers import PreTrainedTokenizerFast
import numpy as np
import pandas as pd
import argparse

# Argument parser for command-line arguments
parser = argparse.ArgumentParser(description="Test Ziggy model in ONNX format")
parser.add_argument('--quant_file', type=str, required=True, help="Path to the Quantized model file")
parser.add_argument('--req_file', type=str, help="Path to the requirements file")
parser.add_argument('--vocab_path', type=str, help="Path to the tokenizer config files")

args = parser.parse_args()

# Define parameters
max_seq_length = 512   # Maximum sequence length

# Define input text for testing
input_text = "Access to client data must be restricted to authorised personnel only."

# Initialise tiktoken tokeniser
tokenizer = PreTrainedTokenizerFast.from_pretrained(args.vocab_path, use_fast=True)

# Define function to tokenise and preprocess text
def encode_text(text, max_length):
    tokens = tokenizer.encode(text.lower())
    if len(tokens) > max_length:
        tokens = tokens[:max_length]  # Truncate
    else:
        tokens += [0] * (max_length - len(tokens))  # Pad
    return tokens

# Load ONNX model
ort_session = ort.InferenceSession(args.quant_file)
print("Input Metadata:")
for input_meta in ort_session.get_inputs():
    print(f"Name: {input_meta.name}, Type: {input_meta.type}, Shape: {input_meta.shape}")

# Load labels and create mappings
labels = pd.read_csv(args.req_file)
id2label = pd.Series(labels.requirement.values, index=labels.id).to_dict()
label2id = pd.Series(labels.id.values, index=labels.requirement).to_dict()
num_classes = len(id2label)

# Tokenise and preprocess
input_ids = torch.tensor([encode_text(input_text, max_seq_length)], dtype=torch.long)
attention_mask = (input_ids != 0).numpy().astype(np.float32)

# Apply softmax to logits to compute probabilities
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

# Predict and compute probabilities
logits = ort_session.run(None, {"input_ids": input_ids.cpu().numpy().astype(np.int64), "attention_mask": attention_mask.astype(np.float32)})[0]

# Compute probabilities and predicted label
probabilities = softmax(logits)
predicted_label = np.argmax(probabilities, axis=1)[0]
predicted_probability = probabilities[0][predicted_label]

# Print the id2label mapping of the predicted label
print(f"Predicted Label: {id2label[predicted_label]} ({predicted_label})")
print(f"Probability: {predicted_probability * 100:.2f}%")
print(f"Probabilities: {probabilities[0]}")