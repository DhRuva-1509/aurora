import torch
from torch import nn
from torch.optim import AdamW
from sklearn.metrics import f1_score
from tqdm import tqdm

from tiny_bert import TinyTransformerClassifier, TinyConfig
from dataloader import get_dataloaders

def train_model(epochs=10, patience=3, device='mps'):
    # Load data
    train_loader, val_loader = get_dataloaders(batch_size=64, model_vocab="bert-base-uncased")

    # Initialize model
    model = TinyTransformerClassifier(TinyConfig).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=3e-4)

    best_val_f1 = 0.0
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Epoch {epoch+1}: Training Loss = {total_loss:.4f} | Validation F1 = {val_f1:.4f}")

        # Early Stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_tiny_bert_model.pt")
            print("Best model saved.")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"No improvement. Patience: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    train_model(device=device)
