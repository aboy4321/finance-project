import torch
import torch.nn as nn
import math 

# embedding code
class TokenEmbedding(nn.Module):
    def __init__(self, size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(size, hidden_dim)

    def forward(self, x):
        return self.embedding(x)

# the code below gives us information about which position each token is in
class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=5000):
        super()__init__()
        
        # creates tensor
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arrange
    


