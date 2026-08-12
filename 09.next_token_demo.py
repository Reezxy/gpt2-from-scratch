import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F

# We reuse the logits shape from the previous step ([1, 10, 50257]).
# Simulated here, because we do not rebuild the GPT-2 model in this file:
torch.manual_seed(0)
logits = torch.randn(1, 10, 50257)  # for real: logits = model(token_ids)

# We only care about the prediction for the next (11th) token,
# i.e. the LAST position in the sequence:
last_logits = logits[0, -1, :]  # Shape: (50257,)
print("Logits for the next token (shape):", last_logits.shape)

# Softmax turns them into real probabilities
probs = F.softmax(last_logits, dim=-1)
print("Sum of all probabilities:", probs.sum().item())  # should be 1.0

# The 5 most likely next token IDs (just numbers without a real tokenizer)
top5_probs, top5_ids = torch.topk(probs, 5)
print("\nTop 5 most likely next token IDs:")
for token_id, prob in zip(top5_ids.tolist(), top5_probs.tolist()):
    print(f"  Token ID {token_id}: {prob:.4%}")

# Deterministic choice: simply take the most likely token
next_token_greedy = torch.argmax(probs).item()
print("\nArgmax (most likely token):", next_token_greedy)

# Random choice weighted by probability (more variety)
next_token_sampled = torch.multinomial(probs, num_samples=1).item()
print("Sampling (rolled according to probability):", next_token_sampled)