# inference_engine.py

import torch
from threading import Thread
from transformers import AutoTokenizer, TextIteratorStreamer
from optimum.intel.openvino import OVModelForCausalLM

MODEL_DIR = r"C:\Users\B-ECE-312-21\phi2_openvino"
TOKENIZER_DIR = r"C:\Users\B-ECE-312-21\phi2_openvino"

# ✅ Load model and tokenizer at the top level
print("🔁 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
model = OVModelForCausalLM.from_pretrained(MODEL_DIR, device="CPU")
print("✅ Model loaded!")

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=1000,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
        streamer=streamer
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for token in streamer:
        yield token
