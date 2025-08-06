import torch
import torch.nn as nn
import numpy as np

# Configuration for TinyBERT
class TinyConfig:
    d_model = 128         # Hidden size
    n_heads = 4           # Number of attention heads
    n_layers = 2          # Number of transformer encoder layers
    ffn_dim = 512         # Feedforward network dimension
    vocab_size = 30522    # BERT vocab size
    max_len = 128         # Maximum sequence length
    num_classes = 5       # Number of emotion classes

# Positional Encoding layer (sinusoidal)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# Tiny Transformer-based Emotion Classifier
class TinyTransformerClassifier(nn.Module):
    def __init__(self, config=TinyConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position = PositionalEncoding(config.d_model, config.max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_dim,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        self.classifier = nn.Linear(config.d_model, config.num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.position(x)
        x = self.encoder(x)
        x = x.mean(dim=1)  # Mean pooling across sequence
        return self.classifier(x)
