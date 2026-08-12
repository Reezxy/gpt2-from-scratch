import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import urllib.request
import torch
import tiktoken

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
data_path = "shakespeare.txt"

if not os.path.exists(data_path):
    urllib.request.urlretrieve(url, data_path)
    print("Downloaded.")
else:
    print("File already exists.")

with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

print("Number of characters in the text:", len(text))
print("\nFirst 300 characters:\n", text[:300])

tokenizer = tiktoken.get_encoding("gpt2")
tokens = tokenizer.encode(text)
print("\nTotal number of tokens:", len(tokens))

data = torch.tensor(tokens, dtype=torch.long)

# 90% training, 10% validation (to check later whether the model really learns instead of memorising)
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

print("\nTrain tokens:", len(train_data))
print("Val tokens:", len(val_data))


def get_batch(data, block_size, batch_size):
    start_indices = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in start_indices])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in start_indices])  # shifted by 1!
    return x, y


x, y = get_batch(train_data, block_size=8, batch_size=4)
print("\nx (input) shape:", x.shape)
print(x)
print("\ny (target, shifted by 1) shape:", y.shape)
print(y)

# For clarity: first example in the batch, token by token
print("\nExample: what should the model predict at each position?")
for t in range(x.shape[1]):
    context = x[0, :t+1].tolist()
    target = y[0, t].item()
    print(f"  Context {context} -> prediction target: {target}")