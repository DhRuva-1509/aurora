import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

# Label encoding
label2id = {"HAPPY": 0, "SAD": 1, "ANGRY": 2, "CONFUSED": 3, "GUILT": 4}

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def get_dataloaders(batch_size=32, max_len=128, model_vocab="bert-base-uncased"):
    # 1. Load processed_emotions.csv
    df = pd.read_csv("processed_emotions.csv")
    df["label_id"] = df["label"].map(label2id)

    # 2. Train/Val split
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label_id"], random_state=42)

    # 3. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_vocab)

    # 4. Wrap into Dataset
    train_dataset = EmotionDataset(train_df["text"].tolist(), train_df["label_id"].tolist(), tokenizer, max_len)
    val_dataset = EmotionDataset(val_df["text"].tolist(), val_df["label_id"].tolist(), tokenizer, max_len)

    # 5. Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader
