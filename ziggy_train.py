# Description: Train Ziggy for multi-label text classification and export it to ONNX format - Custom Tokenizer
# Author: Paul Zanna
# Date: 27/12/2024

# Import Torch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Import ONNX libraries
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, QuantizationMode
from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
from onnxruntime.quantization.registry import IntegerOpsRegistry
import onnxslim

# Import support libraries
import argparse
from transformers import PreTrainedTokenizerFast
import numpy as np
import pandas as pd
import random
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from collections import Counter

# Import Optimum libraries
from optimum.exporters.onnx import main_export, export_models
from optimum.onnx.graph_transformations import check_and_save_model
from optimum.exporters.tasks import TasksManager

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Argument parser for command-line arguments
parser = argparse.ArgumentParser(description="Train Ziggy for multi-label text classification and export it to ONNX format")
parser.add_argument('--model_file', type=str, required=True, help="Path to save the model file")
parser.add_argument('--onnx_file', type=str, required=True, help="Path to save the ONNX model file")
parser.add_argument('--quant_file', type=str, required=True, help="Path to save the Quantized model file")
parser.add_argument('--data_file', type=str, required=True, help="Path to the data file")
parser.add_argument('--vocab_path', type=str, required=True, help="Path to the vocab configuration files")
parser.add_argument('--req_file', type=str, help="Path to the requirements file")

args = parser.parse_args()

tokenizer = PreTrainedTokenizerFast.from_pretrained(args.vocab_path, use_fast=True)

# Define function to tokenise and preprocess text
def encode_text(text, max_length):
    tokens = tokenizer.encode(text.lower())
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    else:
        tokens += [0] * (max_length - len(tokens))
    return tokens

# Hyperparameters
vocab_size = tokenizer.vocab_size   # Tokeniser vocabulary size
embed_dim = 384                     # Embedding dimension
num_heads = 6                       # Number of attention heads
num_layers = 6                      # Number of transformer layers
max_seq_length = 512               # Maximum sequence length
learning_rate = 1e-4                # Learning rate
batch_size = 32                     # Batch size
epochs = 20                         # Number of training epochs
dropout = 0.1                       # Dropout rate

# Quantization-related configs
qmode = QuantizationMode.IntegerOps     # Quantization modes
per_channel = True                      # Enable per-channel quantization
reduce_range = True                     # Use reduced range for 7-bit quantization
block_size = None                       # Not used for Q8 but available for other modes
is_symmetric = True                     # Symmetric quantization
accuracy_level = None                   # Not applicable for Q8
weight_type = QuantType.QInt8           # Weight quantization type
quant_type = QuantType.QUInt8           # Activation quantization type

# Dataset Definition
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, max_length):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label_vector = self.labels[idx]  # e.g. [0, 1, 1, 0, ...]
        input_ids = torch.tensor(encode_text(text, self.max_length), dtype=torch.long)
        # Generate attention mask
        attention_mask = (input_ids != 0).float()  # Aligns with the padding in input_ids

        # Convert label vector to float for BCEWithLogitsLoss
        label_tensor = torch.tensor(label_vector, dtype=torch.float)
        return input_ids, attention_mask, label_tensor

# Model Definition
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, num_classes, max_seq_length):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            batch_first=True, 
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        # input_ids: (batch_size, seq_len)
        # attention_mask: (batch_size, seq_len)
        embedded = self.embedding(input_ids) + self.positional_encoding[:, :input_ids.size(1), :]
        transformer_output = self.transformer_encoder(
            embedded, src_key_padding_mask=~attention_mask.bool()
        )
        # Simple mean-pooling over sequence dimension
        pooled_output = transformer_output.mean(dim=1)
        logits = self.fc(pooled_output)
        return logits

# Training Function
def train_model(model, dataloader, epochs, learning_rate, device):
    model = model.to(device)
    
    # Use BCEWithLogitsLoss for multi-label classification
    loss_fn = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimiser.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()

        # Print average loss after each epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Evaluate after each epoch
        evaluate_model(model, dataloader, device)

# Evaluation Function
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    # Disable gradient computation
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            # Store for metric calculation
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # Concatenate predictions and labels from all batches
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Compute micro-F1
    micro_f1 = f1_score(all_labels, all_preds, average="micro")
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    # Optionally compute exact-match accuracy
    exact_matches = np.all(all_preds == all_labels, axis=1)
    exact_match_accuracy = np.mean(exact_matches)

    # Print evaluation metrics
    print(f"Exact Match Accuracy: {100 * exact_match_accuracy:.2f}%")
    print(f"Micro-F1: {100 * micro_f1:.2f}%")
    print(f"Macro-F1: {100 * macro_f1:.2f}%")

# Load Labels & Data
labels_df = pd.read_csv(args.req_file)
label_columns = labels_df['requirement'].tolist()
num_classes = len(label_columns)

# Load training data
clause_data = pd.read_csv(args.data_file)

# Instead of converting to a single label, keep the multi-label vector:
clause_data['label_vector'] = clause_data[label_columns].values.tolist()
clauses = clause_data['clause'].tolist()
labels_multi = clause_data['label_vector'].tolist()  # Each item is e.g. [0,1,0,...]

# Count how many times this label is '1' across all samples
print("Label distribution in your dataset:")
for idx, label_name in enumerate(label_columns):
    label_count = sum(lv[idx] for lv in labels_multi)
    print(f"{label_name}: {label_count}")

# Build PyTorch Dataset & DataLoader
dataset = TextClassificationDataset(clauses, labels_multi, max_seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize Model
model = TransformerClassifier(
    vocab_size, 
    embed_dim, 
    num_heads, 
    num_layers, 
    num_classes, 
    max_seq_length
)

print("Tokenizer vocab size:", vocab_size)
print("Model vocab size:", model.embedding.num_embeddings)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Train Model
train_model(model, dataloader, epochs, learning_rate, device)
torch.save(model.state_dict(), args.model_file)

# Export to ONNX
dummy_input_ids = torch.randint(0, vocab_size, (1, max_seq_length)).to(device)
dummy_attention_mask = torch.ones(1, max_seq_length).to(device)

# Model export parameters
torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask),
    args.onnx_file,
    export_params=True,
    opset_version=14,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_length"},
        "attention_mask": {0: "batch_size", 1: "seq_length"},
        "logits": {0: "batch_size"},
    }
)

# Verify and Slim
onnx_model = onnx.load(args.onnx_file)
try:
    slimmed_model = onnxslim.slim(onnx_model)
    check_and_save_model(slimmed_model, args.onnx_file)
except Exception as e:
    print(f"Failed to slim model: {e}")
print(f"Slim model saved at {args.onnx_file}")

# Quantize
print("Quantizing the ONNX model to Int8...")
onnx_model = onnx.load(args.onnx_file)

# Model quantization parameters
quantizer = ONNXQuantizer(
    model=onnx_model,
    per_channel=per_channel,
    reduce_range=reduce_range,
    mode=qmode,
    static=False,
    weight_qType=weight_type,
    activation_qType=quant_type,
    tensors_range=None,
    nodes_to_quantize=[],
    nodes_to_exclude=[],
    op_types_to_quantize=list(IntegerOpsRegistry.keys()),
    extra_options=dict(EnableSubgraph=True, MatMulConstBOnly=True),
)

# Quantize the model
quantizer.quantize_model()

# Save the quantized model
check_and_save_model(quantizer.model.model, args.quant_file)
print(f"Quantized model saved at {args.quant_file}")