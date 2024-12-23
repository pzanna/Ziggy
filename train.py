import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import onnx
import onnxruntime as ort
import tiktoken
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from collections import Counter

import onnxslim
from optimum.exporters.onnx import main_export, export_models
from optimum.onnx.graph_transformations import check_and_save_model
from optimum.exporters.tasks import TasksManager
from optimum.onnx.graph_transformations import check_and_save_model
from onnxruntime.quantization import QuantType, QuantizationMode
from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
from onnxruntime.quantization.registry import IntegerOpsRegistry

# Initialise tiktoken tokeniser
tokeniser = tiktoken.get_encoding("gpt2")

def encode_text(text, max_length):
    tokens = tokeniser.encode(text, allowed_special={"<|endoftext|>"})
    if len(tokens) > max_length:
        tokens = tokens[:max_length]  # Truncate
    else:
        tokens += [0] * (max_length - len(tokens))  # Pad
    return tokens

# Define hyperparameters
vocab_size = tokeniser.n_vocab  # Tokeniser vocabulary size
embed_dim = 384
num_heads = 6
num_layers = 6
max_seq_length = 512
learning_rate = 1e-4
batch_size = 32
epochs = 10
dropout = 0.2

# Define file paths
model_path = "/Users/paulzanna/Github/Ziggy/model/"
model_filename = "ziggy_model.bin"
onnx_model_filename = "ziggy_model.onnx"
quant_model_filename = "model_quantized.onnx"
data_path = "/Users/paulzanna/Github/Ziggy/data/"
data_filename = "data.csv"
req_filename = "requirements.csv"

# Define the quantization configuration based on provided arguments
qmode = QuantizationMode.IntegerOps     # Quantization modes
per_channel = True                      # Enable per-channel quantization
reduce_range = True                     # Use reduced range for 7-bit quantization
block_size = None                       # Not used for Q8 but available for other modes
is_symmetric = True                     # Symmetric quantization
accuracy_level = None                   # Not applicable for Q8
weight_type = QuantType.QInt8           # Weight quantization type
quant_type = QuantType.QUInt8           # Activation quantization type

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, max_length):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        input_ids = torch.tensor(encode_text(text, self.max_length), dtype=torch.long)
        attention_mask = (input_ids != 0).long()  # Mask non-padding tokens
        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)
    
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, num_classes, max_seq_length):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids) + self.positional_encoding[:, :input_ids.size(1), :]
        transformer_output = self.transformer_encoder(
            embedded, src_key_padding_mask=~attention_mask.bool()
        )
        pooled_output = transformer_output.mean(dim=1)
        logits = self.fc(pooled_output)
        return logits
    
def train_model(model, dataloader, epochs, learning_rate, device):
    model = model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Training loop
        for input_ids, attention_mask, labels in dataloader:
            input_ids, attention_mask, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                labels.to(device),
            )
            optimiser.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
        
        # Print epoch loss
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Evaluate model accuracy after each epoch
        evaluate_model(model, dataloader, device)


def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids, attention_mask, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                labels.to(device),
            )
            outputs = model(input_ids, attention_mask)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0
    print(f"Validation Accuracy: {accuracy:.2%}")

# Load labels and create mappings
labels = pd.read_csv(data_path + req_filename)
id2label = pd.Series(labels.requirement.values, index=labels.id).to_dict()
label2id = pd.Series(labels.id.values, index=labels.requirement).to_dict()
num_classes = len(id2label)

# print labels
# print(labels)
# print(id2label)
# print(label2id)

# Load data
clause_data = pd.read_csv(data_path + data_filename)

# Combine label columns into a single multi-label 'label' column
label_columns = labels['requirement'].tolist()
clause_data['label'] = clause_data[label_columns].values.tolist()
clause_data = clause_data.drop(columns=label_columns)

# Find which item in clause is the label
clause_data['label'] = clause_data['label'].apply(lambda x: [i for i, v in enumerate(x) if v == 1])
clauses = clause_data['clause'].tolist()
clause_label = clause_data['label'].apply(lambda x: x[0] if x else -1)  # Convert to single integer label or -1 if empty
clause_data = clause_data[clause_data['label'] != -1]  # Remove empty labels
clause_label = clause_label.to_list()

# label distribution
print("Label distribution", dict(Counter(clause_label)))

# Convert clauses to a dataset
dataset = TextClassificationDataset(clauses, clause_label, max_seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialise model
model = TransformerClassifier(vocab_size, embed_dim, num_heads, num_layers, num_classes, max_seq_length)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using " + str(device) + " for training")

# Train model
train_model(model, dataloader, epochs, learning_rate, device)

# Save model
torch.save(model.state_dict(), model_path + model_filename)

# Export ONNX model
dummy_input_ids = torch.randint(0, vocab_size, (1, max_seq_length)).to(device)
dummy_attention_mask = torch.ones(1, max_seq_length).to(device)

torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask),
    model_path + onnx_model_filename,
    opset_version=14,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_length"},
        "attention_mask": {0: "batch_size", 1: "seq_length"},
        "logits": {0: "batch_size"},
    },
    input_types=[torch.int64, torch.int64]
)

# Verify ONNX model
def predict_with_onnx(ort_session, input_ids, attention_mask):
    inputs = {
        "input_ids": input_ids.cpu().numpy().astype(np.int64),  # Ensure int64 type
        "attention_mask": attention_mask.astype(np.int64),
    }
    logits = ort_session.run(None, inputs)[0]
    return np.argmax(logits, axis=1)

onnx_model = onnx.load(model_path + onnx_model_filename)

try:
    slimmed_model = onnxslim.slim(onnx_model)
    check_and_save_model(slimmed_model, model_path + onnx_model_filename)
except Exception as e:
    print(f"Failed to slim {model_path + onnx_model_filename}: {e}")

print(f"Slim model saved at {model_path + onnx_model_filename}")

### Step 4: Quantize the ONNX model if requested
onnxModel = onnx.load(model_path + onnx_model_filename)
print("Quantizing the ONNX model to Int8...")
quantizer = ONNXQuantizer(
    model=onnxModel,
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
    extra_options=dict(
        EnableSubgraph=True,
        MatMulConstBOnly=True,
    ),
)

quantizer.quantize_model()
check_and_save_model(quantizer.model.model, model_path + quant_model_filename)
print(f"Quantized model saved at {model_path + quant_model_filename}")
