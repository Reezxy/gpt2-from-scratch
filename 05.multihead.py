import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)  # mixes the heads back together at the end

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # (batch, seq_len, embed_dim) -> (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)

        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        out = attn_weights @ V  # (batch, num_heads, seq_len, head_dim)

        # Merge the heads back into one vector per word
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        return self.out_proj(out)


# Test
torch.manual_seed(0)
batch_size = 1
seq_len = 3
embed_dim = 8
num_heads = 2

x = torch.randn(batch_size, seq_len, embed_dim)
mha = MultiHeadAttention(embed_dim, num_heads)
output = mha(x)

print("Input (shape):", x.shape)
print("Output (shape):", output.shape)
print(output)