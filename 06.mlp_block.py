import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn

embed_dim = 8

torch.manual_seed(0)
x = torch.randn(1, 3, embed_dim) * 100  # deliberately huge values to make the effect visible
print("x before LayerNorm (first token):", x[0, 0])
print("Mean:", x[0, 0].mean().item(), " Std:", x[0, 0].std().item())

ln = nn.LayerNorm(embed_dim)
x_norm = ln(x)
print("\nx after LayerNorm (first token):", x_norm[0, 0])
print("Mean:", x_norm[0, 0].mean().item(), " Std:", x_norm[0, 0].std().item())


class MLP(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        hidden_dim = embed_dim * 4  # GPT-2 expands 4x internally
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


mlp = MLP(embed_dim)

# Residual connection: original x PLUS the processed (normalised) x
output = x + mlp(ln(x))

print("\nOutput shape (should match x):", output.shape)
print("Output first token:", output[0, 0])