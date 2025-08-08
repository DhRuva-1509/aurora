import pandas as pd
from pathlib import Path
import ast

# === Step 1: Load and combine all CSVs ===
data_dir = Path("data/full_dataset")
dfs = [pd.read_csv(f) for f in data_dir.glob("goemotions_*.csv")]
df = pd.concat(dfs, ignore_index=True)

# === Step 2: Define original 28 emotion columns ===
# Use this order to match your label2id / id2label
EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral"
]

# === Step 3: Convert multi-label → single label (pick first label) ===
def get_primary_label(row):
    for emotion in EMOTIONS:
        if row.get(emotion) == 1:
            return emotion
    return None

df["label"] = df.apply(get_primary_label, axis=1)
df = df[df["label"].notnull()][["text", "label"]].reset_index(drop=True)

# === Step 4: Add negation-aware examples manually ===
negation_examples = [
    ("I'm not happy", "sadness"),
    ("I'm not angry", "neutral"),
    ("I'm not sure how I feel", "confusion"),
    ("I don't love it", "disapproval"),
    ("That wasn't exciting", "disappointment"),
    ("Not feeling proud today", "sadness"),
    ("I’m not mad, just disappointed", "disappointment"),
    ("I’m not thrilled about this", "disapproval"),
    ("Not my favorite thing", "sadness"),
    ("I don’t admire that", "disapproval")
]
neg_df = pd.DataFrame(negation_examples, columns=["text", "label"])
df = pd.concat([df, neg_df], ignore_index=True)

# === Step 5: Save processed file ===
df.to_csv("processed_emotions.csv", index=False)
print(f"✅ Saved {len(df)} rows to processed_emotions.csv with 28-class support + negation samples.")
