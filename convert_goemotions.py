import pandas as pd
from pathlib import Path

# === Define your custom emotion mapping ===
GOEMOTIONS_TO_CUSTOM = {
    "joy": "HAPPY",
    "love": "HAPPY",
    "optimism": "HAPPY",
    "relief": "HAPPY",
    "excitement": "HAPPY",
    "amusement": "HAPPY",
    "admiration": "HAPPY",
    "gratitude": "HAPPY",
    "pride": "HAPPY",

    "sadness": "SAD",
    "grief": "SAD",
    "disappointment": "SAD",

    "anger": "ANGRY",
    "annoyance": "ANGRY",
    "disapproval": "ANGRY",
    "disgust": "ANGRY",

    "confusion": "CONFUSED",
    "realization": "CONFUSED",
    "curiosity": "CONFUSED",

    "remorse": "GUILT",
    "embarrassment": "GUILT"
}

# === Load and combine CSVs ===
data_dir = Path("data/full_dataset")
dfs = [pd.read_csv(f) for f in data_dir.glob("goemotions_*.csv")]
df = pd.concat(dfs, ignore_index=True)

# === Get all GoEmotions label columns ===
emotion_cols = list(GOEMOTIONS_TO_CUSTOM.keys())

# === For each row, find all labels that are 1 ===
def get_custom_label(row):
    for col in emotion_cols:
        if row[col] == 1:
            return GOEMOTIONS_TO_CUSTOM[col]
    return None  # No matching label

df["label"] = df.apply(get_custom_label, axis=1)

# === Drop rows with no matching emotion ===
filtered_df = df[df["label"].notnull()][["text", "label"]].reset_index(drop=True)

# === Save processed file ===
filtered_df.to_csv("processed_emotions.csv", index=False)
print(f"✅ Saved {len(filtered_df)} labeled rows to processed_emotions.csv")
