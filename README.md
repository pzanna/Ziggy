# Ziggy

An ultra small language model for text classification

python3 -m venv ziggy

source ziggy/bin/activate

pip install torch transformers tiktoken onnx onnxruntime pandas datasets scikit-learn

ziggy_train.py --model_file ziggy_model.bin --onnx_file ziggy_model.onnx --quant_file ziggy_model_quantized.onnx --data_file data.csv --req_file requirements.csv --vocab_path /model/

ziggy_test.py --quant_file ziggy_model_quantized.onnx --req_file requirements.csv --vocab_path /model/

ziggy_tokenize.py --word_file word_list.txt --config_path /model/

words.py --text_file input.txt --word_file word_list.txt
