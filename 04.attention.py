import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F

embedding_dim = 4
seq_len = 3

torch.manual_seed(0)
x = torch.randn(seq_len, embedding_dim)  # simulates our 3 word vectors from step 3
print("Input x (shape):", x.shape)

# Three separate linear layers: they turn x into Query, Key and Value
query = nn.Linear(embedding_dim, embedding_dim, bias=False)
key   = nn.Linear(embedding_dim, embedding_dim, bias=False)
value = nn.Linear(embedding_dim, embedding_dim, bias=False)

Q = query(x)
K = key(x)
V = value(x)
print("\nQ, K, V shapes:", Q.shape, K.shape, V.shape)

# Scores: how well does every query match every key? (matrix multiplication over all pairs)
scores = Q @ K.T
print("\nRaw scores (shape):", scores.shape)
print(scores)

# Scaling for more stable gradients (standard in GPT-2)
scores = scores / (embedding_dim ** 0.5)

# Causal mask: position i may only look at positions <= i
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))
print("\nScores after masking:")
print(scores)

# Softmax: scores -> probabilities, every row sums to 1
attn_weights = F.softmax(scores, dim=-1)
print("\nAttention weights:")
print(attn_weights)

# Output: weighted sum of the value vectors
output = attn_weights @ V
print("\nAttention output (shape):", output.shape)
print(output)