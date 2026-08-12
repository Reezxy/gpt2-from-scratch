import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import math
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj.RESIDUAL_SCALE = True  # flag for the special init below

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)

        out = attn_weights @ V
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_proj(out)


class MLP(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        hidden_dim = embed_dim * 4
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.fc2.RESIDUAL_SCALE = True  # flag for the special init below

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class GPTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.ModuleList([GPTBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "RESIDUAL_SCALE", False):
                std = std / math.sqrt(2 * self.num_layers)  # damps the residual output layers
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.head(x)


url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
data_path = "shakespeare.txt"
if not os.path.exists(data_path):
    urllib.request.urlretrieve(url, data_path)

with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")
tokens = tokenizer.encode(text)
data = torch.tensor(tokens, dtype=torch.long)

split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]


def get_batch(data, block_size, batch_size, device):
    start_indices = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in start_indices])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in start_indices])
    return x.to(device), y.to(device)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

block_size = 64
model = GPT2(
    vocab_size=tokenizer.n_vocab,
    embed_dim=128,
    num_heads=4,
    num_layers=4,
    max_seq_len=block_size,
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters (mini model): {num_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

batch_size = 16
num_steps = 500
eval_interval = 25

start_time = time.time()

for step in range(num_steps):
    x, y = get_batch(train_data, block_size, batch_size, device)
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0 or step == num_steps - 1:
        with torch.no_grad():
            xv, yv = get_batch(val_data, block_size, batch_size, device)
            val_logits = model(xv)
            val_loss = F.cross_entropy(val_logits.view(-1, val_logits.size(-1)), yv.view(-1))
        elapsed = time.time() - start_time
        print(f"Step {step:4d} | train loss {loss.item():.4f} | val loss {val_loss.item():.4f} | {elapsed:.1f}s")

torch.save(model.state_dict(), "gpt2_mini.pt")
print("\nModel saved as gpt2_mini.pt")