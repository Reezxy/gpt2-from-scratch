import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import math


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

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.head(x)


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, block_size, device, temperature=0.8):
    model.eval()
    token_ids = tokenizer.encode(prompt)
    token_ids = torch.tensor([token_ids], dtype=torch.long, device=device)  # shape (1, seq_len)

    for _ in range(max_new_tokens):
        # Only use the last block_size tokens as context
        context = token_ids[:, -block_size:]

        logits = model(context)              # (1, seq_len, vocab_size)
        last_logits = logits[0, -1, :]        # only the prediction for the next token
        last_logits = last_logits / temperature

        probs = F.softmax(last_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # shape (1,)

        # Append the new token
        token_ids = torch.cat([token_ids, next_id.unsqueeze(0)], dim=1)

    output_ids = token_ids[0].tolist()
    return tokenizer.decode(output_ids)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")

block_size = 64
model = GPT2(
    vocab_size=tokenizer.n_vocab,
    embed_dim=128,
    num_heads=4,
    num_layers=4,
    max_seq_len=block_size,
).to(device)

model.load_state_dict(torch.load("gpt2_mini.pt", map_location=device))

prompt = "ROMEO:"
result = generate(model, tokenizer, prompt, max_new_tokens=100, block_size=block_size, device=device, temperature=0.8)

print("Generated text:\n")
print(result)