from datasets import load_dataset, concatenate_datasets
from datasets.features import Features, Value
from transformers import AutoTokenizer
import torch

def get_dataloaders(batch_size=32, max_len=128, model_vocab="bert-base-uncased"):
    # 1. Load datasets
    sst2 = load_dataset("glue", "sst2")
    imdb = load_dataset("imdb")

    # 2. Rename IMDB text field to 'sentence'
    imdb["train"] = imdb["train"].rename_column("text", "sentence")
    imdb["test"] = imdb["test"].rename_column("text", "sentence")

    # 3. Normalize SST-2: convert label to int + remove 'idx'
    def map_sst2(example):
        example["label"] = int(example["label"])
        return example

    sst2["train"] = sst2["train"].map(map_sst2)
    sst2["validation"] = sst2["validation"].map(map_sst2)

    sst2["train"] = sst2["train"].remove_columns(["idx"])
    sst2["validation"] = sst2["validation"].remove_columns(["idx"])

    # 👇 Cast SST-2 to Value("int64")
    sst2["train"] = sst2["train"].cast(Features({
        "sentence": Value("string"),
        "label": Value("int64")
    }))
    sst2["validation"] = sst2["validation"].cast(Features({
        "sentence": Value("string"),
        "label": Value("int64")
    }))

    # 4. Normalize IMDB
    def map_imdb(example):
        example["label"] = int(example["label"])
        return example

    imdb["train"] = imdb["train"].map(map_imdb)
    imdb["test"] = imdb["test"].map(map_imdb)

    # Optional: reduce size
    imdb["train"] = imdb["train"].shuffle(seed=42).select(range(25000))
    imdb["test"] = imdb["test"].shuffle(seed=42).select(range(5000))

    # 👇 Cast IMDB to match SST-2
    imdb["train"] = imdb["train"].cast(Features({
        "sentence": Value("string"),
        "label": Value("int64")
    }))
    imdb["test"] = imdb["test"].cast(Features({
        "sentence": Value("string"),
        "label": Value("int64")
    }))

    # 5. Concatenate
    train_dataset = concatenate_datasets([sst2["train"], imdb["train"]])
    val_dataset   = concatenate_datasets([sst2["validation"], imdb["test"]])

    # 6. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_vocab)

    def tokenize(example):
        tokens = tokenizer(
            example["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_len
        )
        tokens["label"] = example["label"]
        return tokens

    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset   = val_dataset.map(tokenize, batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "label"])
    val_dataset.set_format(type="torch", columns=["input_ids", "label"])

    # 7. Return DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader
