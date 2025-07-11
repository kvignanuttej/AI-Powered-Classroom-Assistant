from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.onnx import export
from pathlib import Path

model_name = "microsoft/phi-2"
output_dir = Path("onnx_phi2")

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Export to ONNX
export(
    preprocessor=tokenizer,
    model=model,
    output=output_dir,
    opset=11,
    tokenizer=tokenizer
)
