import torch
import onnxruntime as ort
import tiktoken
import numpy as np
import pandas as pd

# Define file paths
model_path = "/Users/paulzanna/Github/Ziggy/model/"
model_filename = "ziggy_model.bin"
onnx_model_filename = "ziggy_model.onnx"
quant_model_filename = "model_quantized.onnx"
data_path = "/Users/paulzanna/Github/Ziggy/data/"
req_filename = "requirements.csv"

# Define parameters
max_seq_length = 512

# Initialise tiktoken tokeniser
tokeniser = tiktoken.get_encoding("gpt2")

def encode_text(text, max_length):
    tokens = tokeniser.encode(text, allowed_special={"<|endoftext|>"})
    if len(tokens) > max_length:
        tokens = tokens[:max_length]  # Truncate
    else:
        tokens += [0] * (max_length - len(tokens))  # Pad
    return tokens

# Load ONNX model
ort_session = ort.InferenceSession(model_path + quant_model_filename)

# Load labels and create mappings
labels = pd.read_csv(data_path + req_filename)
id2label = pd.Series(labels.requirement.values, index=labels.id).to_dict()
label2id = pd.Series(labels.id.values, index=labels.requirement).to_dict()
num_classes = len(id2label)

# Input text
input_text = "XYZ agrees to encrypt all data in transit for the period of the contract."

# Tokenise and preprocess
input_ids = torch.tensor([encode_text(input_text, max_seq_length)], dtype=torch.int64)
attention_mask = (input_ids != 0).numpy().astype(np.float32)

# Apply softmax to logits to compute probabilities
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

# Predict and compute probabilities
logits = ort_session.run(None, {"input_ids": input_ids.cpu().numpy().astype(np.int64), "attention_mask": attention_mask.astype(np.float32)})[0]

# Apply softmax
probabilities = softmax(logits)

# Get predicted label and probability
predicted_label = np.argmax(probabilities, axis=1)[0]
predicted_probability = probabilities[0][predicted_label]

# Print the id2label mapping of the predicted label
print(f"Predicted Label: {id2label[predicted_label]} ({predicted_label})")
print(f"Probability: {predicted_probability * 100:.2f}%")
print(f"Probabilities: {probabilities[0]}")