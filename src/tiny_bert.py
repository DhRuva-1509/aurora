import torch
import torch.nn as nn
import numpy as np

'''
TinyBERT model configuration and positional encoding.
'''
class TinyConfig:
    d_model = 128
    n_heads = 4
    n_layers = 2
    ffn_dim = 512
    vocab_size = 30522
    max_len = 128
    num_classes = 2

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
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(config.d_model, config.num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.position(x)
        x = self.encoder(x)
        x = x.mean(dim=1)  # Mean pooling
        return self.classifier(x)