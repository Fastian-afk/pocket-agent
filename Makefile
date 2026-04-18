.PHONY: all data train quantize demo eval clean

all: data train quantize

data:
	python generate_data.py

train:
	python train.py

quantize:
	python quantize.py

demo:
	python app.py

eval:
	python eval_harness_contract.py

clean:
	rm -rf lora_adapter/ quantized_model/ __pycache__/
	rm -f train_data.jsonl
