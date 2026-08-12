import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from transformers import GPT2LMHeadModel


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # The real GPT-2 uses a bias on Q/K/V too (unlike our earlier simplified version)
        self.query = nn.Linear(embed_dim, embed_dim, bias=True)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=True)
        self.value = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

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
        self.gelu = nn.GELU(approximate="tanh")  # the real GPT-2 uses this approximation
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


# ---------- Load the real GPT-2 124M weights ----------

print("Loading the real GPT-2 124M from Hugging Face (one-time ~500MB download)...")
hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
hf_state = hf_model.state_dict()

embed_dim = 768
num_heads = 12
num_layers = 12
max_seq_len = 1024
vocab_size = 50257

model = GPT2(vocab_size, embed_dim, num_heads, num_layers, max_seq_len)


def copy_(dst, src):
    assert dst.shape == src.shape, f"Shape mismatch: {dst.shape} vs {src.shape}"
    dst.data.copy_(src)


with torch.no_grad():
    copy_(model.token_embedding.weight, hf_state["transformer.wte.weight"])
    copy_(model.position_embedding.weight, hf_state["transformer.wpe.weight"])

    for i in range(num_layers):
        prefix = f"transformer.h.{i}."
        block = model.blocks[i]

        copy_(block.ln1.weight, hf_state[prefix + "ln_1.weight"])
        copy_(block.ln1.bias,   hf_state[prefix + "ln_1.bias"])

        c_attn_w = hf_state[prefix + "attn.c_attn.weight"].T  # (2304, 768)
        c_attn_b = hf_state[prefix + "attn.c_attn.bias"]      # (2304,)
        q_w, k_w, v_w = c_attn_w.split(embed_dim, dim=0)
        q_b, k_b, v_b = c_attn_b.split(embed_dim, dim=0)
        copy_(block.attn.query.weight, q_w)
        copy_(block.attn.query.bias, q_b)
        copy_(block.attn.key.weight, k_w)
        copy_(block.attn.key.bias, k_b)
        copy_(block.attn.value.weight, v_w)
        copy_(block.attn.value.bias, v_b)

        copy_(block.attn.out_proj.weight, hf_state[prefix + "attn.c_proj.weight"].T)
        copy_(block.attn.out_proj.bias,   hf_state[prefix + "attn.c_proj.bias"])

        copy_(block.ln2.weight, hf_state[prefix + "ln_2.weight"])
        copy_(block.ln2.bias,   hf_state[prefix + "ln_2.bias"])

        copy_(block.mlp.fc1.weight, hf_state[prefix + "mlp.c_fc.weight"].T)
        copy_(block.mlp.fc1.bias,   hf_state[prefix + "mlp.c_fc.bias"])
        copy_(block.mlp.fc2.weight, hf_state[prefix + "mlp.c_proj.weight"].T)
        copy_(block.mlp.fc2.bias,   hf_state[prefix + "mlp.c_proj.bias"])

    copy_(model.ln_final.weight, hf_state["transformer.ln_f.weight"])
    copy_(model.ln_final.bias,   hf_state["transformer.ln_f.bias"])

print("All weights transferred successfully!")

num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params:,}")


# ---------- Generate text ----------

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, block_size, device, temperature=0.8):
    model.eval()
    token_ids = tokenizer.encode(prompt)
    token_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        context = token_ids[:, -block_size:]
        logits = model(context)
        last_logits = logits[0, -1, :] / temperature
        probs = F.softmax(last_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        token_ids = torch.cat([token_ids, next_id.unsqueeze(0)], dim=1)

    return tokenizer.decode(token_ids[0].tolist())


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
tokenizer = tiktoken.get_encoding("gpt2")

prompts = [
    "It was a dark and stormy night",
    "The meaning of life is",
]

for prompt in prompts:
    result = generate(model, tokenizer, prompt, max_new_tokens=80, block_size=max_seq_len, device=device, temperature=0.8)
    print("=" * 60)
    print(result)
    print()