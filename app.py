from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import torch
import json

# ─────────────── Load Adapter + Base Model ────────────────
BASE_MODEL_ID = "google/flan-t5-base"
ADAPTER_PATH = "models/LoRA-T5/data/finetuning/models/LoRA-T5-Advice"   # ← Path to your saved LoRA model

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_ID)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).to("mps")

# ─────────────── Inference Function ────────────────
def generate_advice(text: str, sentiment: str) -> str:
    instruction = f"Text: {text}\nSentiment: {sentiment}"
    inputs = tokenizer(instruction, return_tensors="pt").to("mps")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            top_p=0.95,
            temperature=0.4,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ─────────────── Sample Test ────────────────
if __name__ == "__main__":
    text = "I am feeling lonely"
    sentiment = "lonely"
    #expected = "Try joining local clubs or classes that interest you — it’s a great way to meet people and feel less isolated."
    generated = generate_advice(text, sentiment)

    print("\n--- Prediction ---")
    print(f"Input:\nText: {text}\nSentiment: {sentiment}")
    #print(f"Ground Truth Advice:\n{expected}")
    print(f"Generated Advice:\n{generated}")
