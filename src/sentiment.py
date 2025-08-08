import torch
from transformers import AutoTokenizer
from src.tiny_bert import TinyTransformerClassifier, TinyConfig

# Label mappings
label2id = {
    "admiration": 0, "amusement": 1, "anger": 2, "annoyance": 3, "approval": 4,
    "caring": 5, "confusion": 6, "curiosity": 7, "desire": 8, "disappointment": 9,
    "disapproval": 10, "disgust": 11, "embarrassment": 12, "excitement": 13, "fear": 14,
    "gratitude": 15, "grief": 16, "joy": 17, "love": 18, "nervousness": 19,
    "optimism": 20, "pride": 21, "realization": 22, "relief": 23, "remorse": 24,
    "sadness": 25, "surprise": 26, "neutral": 27
}
id2label = {v: k for k, v in label2id.items()}

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = TinyTransformerClassifier(TinyConfig).to('mps')
model.load_state_dict(torch.load("src/best_tiny_bert_model.pt", map_location='mps'))
model.eval()

'''
Function to predict emotion of a given text using the TinyBERT model.
'''
def predict_sentiment(text: str) -> str:
    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=TinyConfig.max_len
    )

    with torch.no_grad():
        input_ids = inputs['input_ids'].to('mps')
        logits = model(input_ids)
        prediction = torch.argmax(logits, dim=1).item()

    return id2label[prediction]
