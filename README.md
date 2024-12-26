# Ziggy

An ultra small language model for text classification

python3 -m venv ziggy
source ziggy/bin/activate
pip install torch tiktoken onnx onnxruntime pandas datasets scikit-learn

train.py --model_file ziggy_model.bin --onnx_file ziggy_model.onnx --quant_file ziggy_model_quantized.onnx --data_file data.csv --req_file requirements.csv
test.py --quant_file ziggy_model_quantized.onnx --req_file requirements.csv
