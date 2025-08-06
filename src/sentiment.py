import torch
from transformers import AutoTokenizer
from src.tiny_bert import TinyTransformerClassifier, TinyConfig

# Label mapping
id2label = {
    0: "HAPPY",
    1: "SAD",
    2: "ANGRY",
    3: "CONFUSED",
    4: "GUILT"
}

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
