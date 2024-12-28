# Ziggy

## Overview

Ziggy is a multi-label text classification system designed for document review and assessment tasks. It features custom tokenization tailored to domain-specific requirements and leverages state-of-the-art deep learning libraries to achieve efficient and accurate text classification using only 14M parameters (can be modified). Ziggy exports models to ONNX format for web deployment, including optional quantization for optimized inference. It is intended for developers and researchers who require flexible tokenization and robust multi-label classification capabilities.

---

## Key Features

- **Custom Tokenizer Integration**:

  - Utilizes a domain-specific tokenizer for precise text processing and input preparation.
  - The custom tokenizer is generated using the `ziggy_tokenize.py` script and is adaptable to specialized vocabularies.

- **Multi-Label Text Classification**:

  - Supports multi-label classification where each input text can be associated with multiple labels.
  - Includes metrics like Exact Match Accuracy and F1-Score for performance evaluation.

- **ONNX Model Export**:

  - Converts the trained PyTorch model to ONNX format for deployment.
  - Generates both a full ONNX model and a quantized (int8) version for optimized inference.

- **Data Processing**:

  - Uses `datasets` and `pandas` for efficient data handling.
  - Supports train-test splitting, data augmentation, and balancing techniques.

- **Evaluation Metrics**:
  - Implements Exact Match Accuracy, Micro-F1, and Macro-F1 for multi-label performance tracking.
  - Outputs detailed performance logs during training and testing.

---

## Libraries Used

- **Deep Learning**: `torch`, `torch.nn`
- **ONNX Optimization**: `onnx`, `onnxruntime`, `optimum`
- **Data Handling**: `pandas`, `numpy`, `datasets`
- **Metrics**: `sklearn.metrics`, `collections.Counter`

---

## How It Works

### 1. Data Preparation

- The script loads text data from the specified dataset and splits it into training and testing sets.
- A custom tokenizer processes the text into token IDs for model consumption.
- Supports preprocessing steps such as:
  - Removing stop words.
  - Cleaning punctuation.
  - Normalizing text (e.g., lowercasing, stripping whitespace).

### 2. Model Training

- A PyTorch-based model architecture is constructed with support for multi-label objectives.
- The training process optimizes the model using metrics like Binary Cross-Entropy loss.
- Training includes:
  - Regular evaluation on the validation set.
  - Learning rate scheduling.
  - Gradient clipping for stability.

### 3. Model Evaluation

- Metrics such as Exact Match Accuracy, Micro-F1, and Macro-F1 are calculated during and after training.
- Evaluates the model's ability to predict all correct labels for a given input.

### 4. ONNX Model Export

- After training, the model is exported to ONNX format using the `optimum` library.
- Two ONNX models are generated:
  - Full precision ONNX model.
  - Quantized (int8) ONNX model for size and speed optimization.

---

## Usage

### Installation

```
pip install torch transformers onnx onnxruntime onnxslim optimum pandas numpy datasets scikit-learn
```

### Training the Model

Run the ziggy_train.py script with the required arguments for dataset paths and configurations.

### Training the Model

Run the ziggy_test.py script with the required arguments to validate the model.

### ONNX Export

After training the script automatically exports the model to two ONNX formats:

- ziggy_model.onnx (full precision)
- ziggy_model_quantized.onnx (int8 quantized)

### Inference

Use the exported ONNX model with ONNX Runtime for efficient deployment.

```
const session = await ort.InferenceSession.create('path/to/ziggy_model_quantized.onnx');
const inputIds = new ort.Tensor("int64", BigInt64Array.from([token_ids]), [1, seq_length]);
const attentionMask = new ort.Tensor("float32", Float32Array.from([mask_values]), [1, seq_length]);
const feeds = { input_ids: inputIds, attention_mask: attentionMask };
const output = await session.run(feeds);
console.log(output);
```

### Tokenizer

The model uses a custom tokenizer and vocabulary which is created using ziggy_tokenize.py

### File Structure

- [ziggy_train.py](../docs/ziggy_train.md): Script for training the model.
- [ziggy_test.py](../docs/ziggy_test.md): Script for testing and validating the model.
- [ziggy_tokenize.py](../docs/ziggy_tokenize.md): Script for creating a custom tokenizer and vocabulary.
- [words.py](../docs/words.md): Script for creating a word file for the custom tokenizer.
