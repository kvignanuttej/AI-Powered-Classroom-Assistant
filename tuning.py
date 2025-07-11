import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer and model (FP16/FP32 version for simulation)
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Load tuning dataset
with open("tuning_data.json", "r") as f:
    tuning_data = json.load(f)

print("Generating fine-tuned-style outputs for dataset...")

for idx, item in enumerate(tuning_data):
    instruction = item["instruction"]
    reference = item["response"]

    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the actual model-generated response (after the instruction)
    response = result.split("### Response:\n")[-1].strip()

    print(f"\n🔹 Instruction: {instruction}")
    print(f"✅ Reference Response: {reference}")
    print(f"🤖 Model Response: {response}")
