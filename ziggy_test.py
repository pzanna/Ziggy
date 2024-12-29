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

def encode_text(text, tokenizer, max_length):
    tokens = tokenizer.encode(text.lower())
    return tokens[:max_length] + [0] * (max_length - len(tokens))

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def main(quant_file, labels_file, vocab_path):
    max_seq_length = 512
    input_text = "Cat"
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(vocab_path, use_fast=True)
    ort_session = ort.InferenceSession(quant_file)
    
    print("Input Metadata:")
    for input_meta in ort_session.get_inputs():
        print(f"Name: {input_meta.name}, Type: {input_meta.type}, Shape: {input_meta.shape}")

    labels = pd.read_csv(labels_file)
    id2label = pd.Series(labels.label.values, index=labels.id).to_dict()
    num_classes = len(id2label)

    input_ids = torch.tensor([encode_text(input_text, tokenizer, max_seq_length)], dtype=torch.long)
    attention_mask = (input_ids != 0).numpy().astype(np.float32)

    logits = ort_session.run(None, {"input_ids": input_ids.cpu().numpy().astype(np.int64), "attention_mask": attention_mask})[0]
    probabilities = softmax(logits)
    predicted_label = np.argmax(probabilities, axis=1)[0]
    predicted_probability = probabilities[0][predicted_label]

    print(f"Number of classes: {num_classes}")
    print(f"Predicted Label: {id2label[predicted_label]} ({predicted_label})")
    print(f"Probability: {predicted_probability * 100:.2f}%")
    print(f"Probabilities: {probabilities[0]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Ziggy model in ONNX format")
    parser.add_argument('--quant_file', type=str, required=True, help="Path to the Quantized model file")
    parser.add_argument('--labels_file', type=str, help="Path to the labels file")
    parser.add_argument('--vocab_path', type=str, help="Path to the tokenizer config files")
    args = parser.parse_args()

    main(args.quant_file, args.labels_file, args.vocab_path)
