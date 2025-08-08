from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from datasets import load_dataset
import torch

# ─── Load Base Model and Tokenizer ─────────────────────────────
model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# ─── Apply LoRA Adapter ────────────────────────────────────────
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)
model = get_peft_model(base_model, peft_config)

# ─── Load and Slice Dataset ────────────────────────────────────
dataset = load_dataset(
    "json",
    data_files={
        "train": "../preprocessing/empatheticdialogues/empathetic_dialogues_advice.jsonl",
        "validation": "../preprocessing/empatheticdialogues/empathetic_dialogues_advice_valid.jsonl"
    }
)
# Reduce training set size to 5000
dataset["train"] = dataset["train"].select(range(5000))

# ─── Preprocess ────────────────────────────────────────────────
def preprocess(example):
    inputs = tokenizer(example["instruction"], padding="max_length", truncation=True, max_length=512)
    targets = tokenizer(example["output"], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess, batched=True)

# ─── Training Arguments ────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./flan-lora-advice",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    save_total_limit=1,
    report_to="none",
    push_to_hub=False
)

# ─── Trainer ───────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

# ─── Train ─────────────────────────────────────────────────────
trainer.train()

# ─── Save Model ────────────────────────────────────────────────
model.save_pretrained("models/LoRA-T5-Advice")
tokenizer.save_pretrained("models/LoRA-T5-Advice")

# ─── Inference Function ────────────────────────────────────────
def generate_advice(text, sentiment):
    input_text = f"Text: {text}\nSentiment: {sentiment}"
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    model = PeftModel.from_pretrained(model, "models/LoRA-T5-Advice")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    inputs = tokenizer(input_text, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ─── Example Usage ─────────────────────────────────────────────
if __name__ == "__main__":
    print(generate_advice("I moved to a new city and feel isolated.", "lonely"))
