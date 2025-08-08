import pandas as pd
import json
from tqdm import tqdm
import re

# Load CSV
train_df = pd.read_csv("../empatheticdialogues/train.csv")
valid_df = pd.read_csv("../empatheticdialogues/valid.csv")

# Keep only listener responses
train_df = train_df[train_df["utterance_idx"] == 1].reset_index(drop=True)
valid_df = valid_df[valid_df["utterance_idx"] == 1].reset_index(drop=True)

def is_good_advice(utterance, context):
    # Filter out similar/duplicate phrases
    if utterance.strip().lower() == context.strip().lower():
        return False
    if utterance.lower().startswith("hi") or utterance.lower().startswith("hello"):
        return False
    if len(utterance.strip()) < 8:
        return False
    # Heuristic: contains advice-like phrasing
    advice_phrases = [
        "you should", "try", "maybe", "i recommend", "it's okay to", "consider", 
        "don't forget", "take time to", "remember to", "it helps to", "talk to"
    ]
    return any(p in utterance.lower() for p in advice_phrases)

def to_instruction_output(row):
    context = row["context"]
    sentiment = row["prompt"]
    utterance = row["utterance"]

    if is_good_advice(utterance, context):
        return {
            "instruction": f"Text: {context}\nSentiment: {sentiment}\nAdvice:",
            "output": utterance.strip()
        }
    return None

# Clean dataset
def clean(df):
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        item = to_instruction_output(row)
        if item:
            results.append(item)
    return results

# Apply
train_cleaned = clean(train_df)
valid_cleaned = clean(valid_df)

# Save to JSONL
with open("clean_flan_t5_train.jsonl", "w") as f:
    for item in train_cleaned:
        f.write(json.dumps(item) + "\n")

with open("clean_flan_t5_valid.jsonl", "w") as f:
    for item in valid_cleaned:
        f.write(json.dumps(item) + "\n")

print(f"✅ Saved: {len(train_cleaned)} train and {len(valid_cleaned)} valid advice-only samples.")
