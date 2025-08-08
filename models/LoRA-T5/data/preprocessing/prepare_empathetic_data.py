import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

# Load FLAN-T5 tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# Load CSVs
train_df = pd.read_csv("../empatheticdialogues/train.csv")
valid_df = pd.read_csv("../empatheticdialogues/valid.csv")

# Keep only utterance_idx == 1
train_df = train_df[train_df["utterance_idx"] == 1].reset_index(drop=True)
valid_df = valid_df[valid_df["utterance_idx"] == 1].reset_index(drop=True)

# Define formatting function for T5 input
def format_for_t5(row):
    input_text = f"Text: {row['prompt']}\nSentiment: {row['context']}"
    target_text = row['utterance']

    input_ids = tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    ).input_ids[0].tolist()

    labels = tokenizer(
        target_text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    ).input_ids[0].tolist()

    return {
        "input_ids": input_ids,
        "labels": labels
    }

# Process with tqdm
tqdm.pandas(desc="Processing training data")
train_processed = train_df.progress_apply(format_for_t5, axis=1, result_type="expand")
valid_processed = valid_df.progress_apply(format_for_t5, axis=1, result_type="expand")

# Save as JSONL
train_df["input_ids"] = train_processed["input_ids"]
train_df["labels"] = train_processed["labels"]
valid_df["input_ids"] = valid_processed["input_ids"]
valid_df["labels"] = valid_processed["labels"]

train_df[["input_ids", "labels"]].to_json("flan_t5_train.jsonl", orient="records", lines=True)
valid_df[["input_ids", "labels"]].to_json("flan_t5_valid.jsonl", orient="records", lines=True)
