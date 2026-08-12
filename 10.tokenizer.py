import tiktoken

# "gpt2" loads exactly the encoding rules the real GPT-2 used
tokenizer = tiktoken.get_encoding("gpt2")

text = "I am rebuilding my own GPT-2 from scratch, and it is fun!"

token_ids = tokenizer.encode(text)
print("Text:", text)
print("\nToken IDs:", token_ids)
print("Number of tokens:", len(token_ids))

# Show every single token piece (to understand how the text was split)
print("\nIndividual token pieces:")
for tid in token_ids:
    piece = tokenizer.decode([tid])
    print(f"  {tid} -> '{piece}'")

# Back to text
decoded = tokenizer.decode(token_ids)
print("\nDecoded back:", decoded)

print("\nTotal vocabulary size:", tokenizer.n_vocab)  # should be 50257