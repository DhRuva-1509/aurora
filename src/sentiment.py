import torch
from transformers import AutoTokenizer
from src.tiny_bert import TinyTransformerClassifier, TinyConfig



tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = TinyTransformerClassifier(TinyConfig).to('mps')
model.load_state_dict(torch.load("best_tiny_bert_model.pt", map_location='mps'))
model.eval()

'''
Function to predict sentiment of a given text using the TinyBERT model.
'''
def predict_sentiment(text: str) -> str:
    inputs = tokenizer(
        text,
        return_tensors = 'pt',
        padding = 'max_length',
        truncation = True,
        max_length = TinyConfig.max_len
    )

    with torch.no_grad():
        logits = model(inputs['input_ids'].to('mps'))
        predictions = torch.argmax(logits, dim=1)
    
    return "Postitive" if predictions==1 else "Negative"
