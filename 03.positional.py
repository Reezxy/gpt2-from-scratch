import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn

vocab_size = 10
embedding_dim = 4
max_seq_len = 8  # longest sentence we want to support

# Two separate embedding tables:
token_embedding = nn.Embedding(vocab_size, embedding_dim)      # "what" the token is
position_embedding = nn.Embedding(max_seq_len, embedding_dim)  # "where" the token stands

token_ids = torch.tensor([3, 7, 1])  # "i like cats"
positions = torch.arange(len(token_ids))  # [0, 1, 2] -> position of every token in the sentence
print("Positions:", positions)

tok_vecs = token_embedding(token_ids)
pos_vecs = position_embedding(positions)

print("\nToken vectors (shape):", tok_vecs.shape)
print("Position vectors (shape):", pos_vecs.shape)

# GPT-2 simply adds the two together:
combined = tok_vecs + pos_vecs
print("\nCombined vectors (shape):", combined.shape)
print(combined)