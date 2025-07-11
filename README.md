# AI-Powered-Classroom-Assistant
This project presents an AI-powered educational assistant built using the OpenVINO™ Toolkit and a Phi-2 language model quantized to INT4 precision. The model is fine-tuned for answering student-specific educational queries, offering fast and optimized performance on Intel hardware (CPU/Iris Xe GPU).



Features:
->Based on Phi-2 language model (by Microsoft).

->Optimized using INT4 quantization for fast inference.

->Fine-tuned with 30 educational prompts (JSON format).

->Deployable on Windows with CPU/GPU.

->Accepts text queries and generates accurate answers.

To implement the project,
->Make sure to download python(3.10+ recommended).

create a python environment in a folder to store all the project's data.
In bash:
python -m venv openvino_env
source openvino_env/bin/activate  # Linux
openvino_env\Scripts\activate     # Windows

->The above commands creates a python environment.

->Download all the required tools from requirements.txt through bash inside the environment.

-> Download Microsoft phi-2 model from hugghing face.
huggingface link: https://huggingface.co/microsoft/phi-2/tree/main.

->make sure to download the the following files for proper implementation of model:
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
model.safetensors.index.json
special_tokens_map.json
tokenizer.json
tokenizer_config.json
vocab.json
config.json
generation_config.json
merges.txt
added_tokens.json

->After downloading the model, fine tuning it to give outputs to educational queries is the next step.

->Run tuning.py to fine tune the model.

->For inferencing, the safetensors model that we tuned should be converted into CPU supported format with the help of Intel openvino toolkit.

->In this process, the safetensors file gets converted to ONNX model which after gets converted to openvino format i.e, an XML and bin file of the models.

->Run phi2_convert_to_onnx.py to convert the safetensors model to onnx model which is suitable for further conversion.

->Run the following command in command prompt to convert the onnx model to xml and bin file using model optimizer.
  mo --input_model onnx_phi2/model.onnx --output_dir phi2_openvino --data_type FP32.
-> This converts the model into an xml and bin file of 32 bit.

-> Using a 32 bit model for inference may become challenging for low end PC's.

->To compress the model from 32 bit to INT8, use NNCF(Neural Network Compression Framework).

->Run calibrate.py, It calibrates the model with some spectific data to compress the model with a verhy minimal accuracy loss.

->INT8 can be further optimized to INT4 for better speeds and minimal loss(if needed).

->Run inference_engine.py to check if the model is working perfectly.

->After running the inference_engine.py successfully, Run app.py that launches UI for the built model.

->app.py internally calls inference_engine.py for inferencing the model.

->This completes the implementation of AI assistant that can be useful for classroom based appliactions.















