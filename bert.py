import math
import torch
import torch.nn as nn

"""
size: numbers of unique tokens to be produced
hidden_dim: dimensionality of embedding vector
x: batch of token ids
"""
class TokenEmbedding(nn.Module):
    def __init__(self, size, hidden_dim):
        super().__init__()
        # creating lookup table
        self.embedding = nn.Embedding(size, hidden_dim)

    def forward(self, x):
        return self.embedding(x)

"""
pe: tensor initialized with zeros
position: creates a column vector
div_term: defines frequency of sine and cosine waves
"""
class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arrange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arrange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position, div_term)
        pe[:, 1::2] = torch.cos(position, div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

"""
self.d_k: the dimensions per heads
self.qkv: produces query, key, and value vector
"""
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim, heads):
        super().__init__()
        self.d_k = hidden_dim // heads

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size()
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.heads, self.d_k)
        q, k, v = qkv.unbind(dim=2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmmul(attn, v)

        context = context.transpose(1, 2).reshape(batch_size. seq_len, hidden_dim)
        return self.out(context)


