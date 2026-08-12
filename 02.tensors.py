import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn

# Imagine a tiny vocabulary: 10 possible tokens (IDs 0-9)
vocab_size = 10
embedding_dim = 4  # every token becomes a vector of 4 numbers

# Embedding table: one row per token, randomly initialised
embedding = nn.Embedding(vocab_size, embedding_dim)
print("Embedding table (shape):", embedding.weight.shape)
print(embedding.weight)

# An example sentence as token IDs, e.g. "i like cats" -> [3, 7, 1]
token_ids = torch.tensor([3, 7, 1])
print("\nToken IDs:", token_ids, "Shape:", token_ids.shape)

# Lookup: every ID gets replaced by its vector
vectors = embedding(token_ids)
print("\nCorresponding vectors (shape):", vectors.shape)
print(vectors)