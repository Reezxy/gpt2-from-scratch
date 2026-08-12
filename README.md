<div align="center">

# 🧠 GPT-2 from Scratch — 17 Files, No Magic

**A learning project: GPT-2 rebuilt from zero in plain PyTorch — one small file per concept — ending with OpenAI's real 124M weights loaded into the hand-written model.**

No `nn.TransformerBlock`. No framework hiding the math. Just embeddings, attention, and a loop — built to be read and re-run by anyone getting into machine learning.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Runs on](https://img.shields.io/badge/runs%20on-MacBook%20(MPS)%20%7C%20CPU%20%7C%20CUDA-lightgrey)](#hardware)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#contributing)

*Free to use, fork and learn from. If it helped you understand transformers, a ⭐ is the only thanks this repo takes.*

</div>

---

## About this project

This is my **learning project** — and I built it so that others can learn from it too.

I rebuilt **GPT-2 from scratch, entirely by myself**: every layer written by hand in plain PyTorch, starting from a single embedding table and ending with a complete language model. As the final step, I **loaded the real GPT-2 weights from Hugging Face into my own model classes** — same architecture, original 124M weights, real English coming out. That is the proof that the from-scratch implementation is genuinely correct and not just "close enough".

Instead of collapsing everything into one polished library, I kept **every step as its own small, runnable file**. That way the repository doubles as a course you can walk through: run one file, read its printed tensor shapes, change a number, run it again.

**If you want to get into machine learning, this repo is for you.** It is meant for developers who know a bit of Python, have heard of transformers, and want to actually understand what happens inside a language model — layer by layer, with nothing hidden behind a framework abstraction. Work through the 17 steps in order and you will have built GPT-2 yourself by the end.

> Most "build GPT" tutorials hand you a finished 500-line file. This one hands you the 17 steps that produce it.

---

## The 30-second version

```bash
git clone https://github.com/Reezxy/gpt2-from-scratch.git
cd gpt2-from-scratch
pip install -r requirements.txt

python 04.attention.py        # watch attention happen on 3 words
python 12_train.py            # train a 7M-param GPT-2 on Shakespeare (~2 min)
python 14.generate.py         # make it talk
python 17_load_real_gpt2.py   # load the REAL GPT-2 124M into your own code
```

That's it. No config system, no CLI framework, no `src/`. Every file runs on its own.

---

## The journey: 17 steps from a tensor to a language model

| # | File | What you build | New idea |
|---|------|----------------|----------|
| 01 | `01_setup.py` | Device check | CPU / Apple MPS / CUDA |
| 02 | `02.tensors.py` | Token embeddings | Words become vectors |
| 03 | `03.positional.py` | Positional embeddings | The model learns *where* a word stands |
| 04 | `04.attention.py` | Single-head attention | Q · Kᵀ, scaling, causal mask, softmax |
| 05 | `05.multihead.py` | Multi-head attention | Splitting into heads with `view` + `transpose` |
| 06 | `06.mlp_block.py` | LayerNorm + MLP | GELU, 4× expansion, residual connection |
| 07 | `07_transformer.block.py` | The transformer block | Pre-norm, two sublayers |
| 08 | `08.gpt2_model.py` | The full GPT-2 | 12 blocks, weight tying — **124,439,808 params** |
| 09 | `09.next_token_demo.py` | Next-token prediction | Logits → softmax → argmax vs. sampling |
| 10 | `10.tokenizer.py` | The real GPT-2 tokenizer | BPE via `tiktoken`, 50,257 tokens |
| 11 | `11.data_prep.py` | Batching | `x` and `y` shifted by exactly one token |
| 12 | `12_train.py` | **First real training run** | AdamW, cross-entropy, scaled residual init |
| 14 | `14.generate.py` | Autoregressive sampling | Context window, temperature |
| 15 | `15_scale_up_train.py` | **Scaling up** | 15 books, 30M params, checkpointing |
| 16 | `16_generate_scaled.py` | Sampling the bigger model | Same code, better output |
| 17 | `17_load_real_gpt2.py` | **The finale** | Port OpenAI's real 124M weights into *your* classes |

Each file is standalone and re-declares the classes it needs. That's duplication on purpose: you can open **any** file cold and read it top to bottom without jumping between imports.

---

## The architecture, in one picture

```
      token IDs  [50257 possible]
           │
   ┌───────▼────────┐        ┌──────────────────┐
   │ token embedding│   +    │ position embedding│
   └───────┬────────┘        └──────────────────┘
           │
     ╔═════▼══════════════════════════════════╗
     ║  transformer block   × N layers        ║
     ║                                        ║
     ║   x = x + Attention(LayerNorm(x))      ║   ← talk between tokens
     ║   x = x + MLP(LayerNorm(x))            ║   ← think about each token
     ╚═════╤══════════════════════════════════╝
           │
      LayerNorm
           │
     Linear → logits  (vocab_size)   ← weights tied to the token embedding
           │
        softmax → next token
```

**The causal mask is the whole trick.** Position `i` may only attend to positions `≤ i`, so the model can be trained on an entire sentence at once while never seeing the future:

```python
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))
```

---

## Three models in this repo

| | **Mini** | **Scaled** | **Real GPT-2 124M** |
|---|---|---|---|
| Params | 7.2M | 30.0M | 124.4M |
| Layers × heads | 4 × 4 | 6 × 6 | 12 × 12 |
| Embedding dim | 128 | 384 | 768 |
| Context | 64 | 256 | 1024 |
| Data | Tiny Shakespeare | 15 Gutenberg books (~9.2M chars) | WebText (OpenAI's) |
| Trained by | you, ~2 min | you, ~1–2 h on a laptop | OpenAI, 2019 |
| Script | `12_train.py` | `15_scale_up_train.py` | `17_load_real_gpt2.py` |

Full hyperparameters: [`configs/model_configs.json`](configs/model_configs.json) · Full data sources: [`data/datasets.json`](data/datasets.json)

---

## Training data

Nothing in this repo is scraped and nothing is committed to git — the scripts download everything on first run, all of it **public domain**.

- **Tiny Shakespeare** (1.1M chars) — the classic first dataset, small enough that the loss visibly drops within a minute.
- **15 Project Gutenberg books** (9.2M chars) — *Pride and Prejudice*, *Frankenstein*, *Moby Dick*, *Dracula*, *Crime and Punishment*, *Alice in Wonderland*, and ten more. Downloaded, boilerplate-stripped, concatenated, BPE-encoded.

The complete manifest — every book ID, URL, license, preprocessing step, split ratio and sample prompt — lives in **[`data/datasets.json`](data/datasets.json)**, so you can see exactly what the model ate without running a single line.

Want your own corpus? Drop `.txt` files into `books/`, delete `book_corpus.txt`, and re-run step 15.

---

## The finale: real weights in hand-written code

The most satisfying moment in the whole project is `17_load_real_gpt2.py`. It downloads OpenAI's original GPT-2 and copies all 124M weights, tensor by tensor, into the classes built in steps 02–08:

```python
c_attn_w = hf_state[prefix + "attn.c_attn.weight"].T   # (2304, 768)
q_w, k_w, v_w = c_attn_w.split(embed_dim, dim=0)       # one fused matrix → Q, K, V
copy_(block.attn.query.weight, q_w)
```

Every copy is guarded by a shape assertion — if your architecture were wrong by a single dimension, it would crash instead of quietly producing noise. It doesn't crash. It generates English.

Three details that must be right, and that most from-scratch tutorials skip:

1. **Hugging Face stores GPT-2 weights transposed** (it uses `Conv1D`, not `nn.Linear`) → every weight needs `.T`.
2. **Q, K and V live in one fused `c_attn` matrix** → split it into three chunks of `embed_dim`.
3. **Real GPT-2 uses the tanh approximation of GELU** and **biases on Q/K/V** → `nn.GELU(approximate="tanh")`, `bias=True`.

The full key-by-key mapping is documented in [`configs/model_configs.json`](configs/model_configs.json) under `weight_mapping`.

---

## Hardware

Written and trained on an **Apple Silicon MacBook** using the MPS backend — no GPU cluster, no cloud bill. Every script auto-selects the device:

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

Running on NVIDIA? Change that one line to `"cuda"`. Running on CPU? It still works, just slower — steps 01–11 are instant everywhere.

> The `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` at the top of each file is a macOS OpenMP workaround. Harmless on other platforms.

---

## What you'll actually understand afterwards

- Why attention is *three* matrices and not one
- Why the scores get divided by `√head_dim`
- What "causal" really means, and why one triangular mask replaces an entire loop
- Why LayerNorm sits *before* the sublayer in GPT-2 (pre-norm) and what breaks without it
- Why the output layer shares weights with the token embedding (weight tying saves 38M params in GPT-2 124M)
- Why residual output layers get initialised with `std = 0.02 / √(2·n_layers)`
- What `temperature` actually does to a probability distribution
- Why the training target is just the input shifted by one token

---

## FAQ

**Is this production code?**
No. It's *teaching* code — deliberately simple, deliberately duplicated across files. There's no KV cache, no flash attention, no mixed precision, no gradient accumulation, no LR schedule. Adding them is the perfect next exercise.

**Where is step 13?**
Folded into steps 12 and 15 — it was the proper GPT-2 weight initialisation, which now lives inside `_init_weights()` in both training scripts.

**Why the inconsistent filenames (`04.attention.py` vs `12_train.py`)?**
Because this repo is an honest record of a learning process, not a cleaned-up rewrite. The numbers are what matter: run them in order.

**Can I train this on my own text?**
Yes — that's the point. Swap the corpus in step 15, keep the tokenizer, adjust `block_size` and `embed_dim` to your patience.

**Will the mini model write good text?**
Absolutely not. 7M parameters on 300k tokens produces Shakespeare-flavoured gibberish — and watching *why* it's gibberish teaches you more than a perfect model would.

---

## How to learn from this repo

**New to machine learning?** Run `01` → `11` in order, read every print statement, change the numbers and run them again. Nothing before step 12 takes longer than a second, and every file prints its tensor shapes at each stage — following the shapes is the fastest way to build real intuition. Then train your own model in step 12 and let it talk in step 14.

**Already comfortable with PyTorch?** Read `08.gpt2_model.py` and `17_load_real_gpt2.py`, then start hacking.

Take your time with step 04. Attention is the one idea everything else in the repo is built on.

---

## Contributing

Issues and PRs are welcome — especially:

- Clearer explanations in the comments
- A KV cache for `generate()` (step 18, anyone?)
- Top-k / top-p sampling
- A cosine LR schedule with warmup
- Notebook versions of the steps

Keep the spirit: **one concept per file, readable top to bottom, no hidden abstractions.**

---

## Credits

Standing on the shoulders of [Andrej Karpathy's](https://github.com/karpathy) nanoGPT and his "Let's build GPT" lecture, the original [GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), and [Project Gutenberg](https://www.gutenberg.org/) for the books.

Model weights in step 17: [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) on Hugging Face.

## License

MIT — see [LICENSE](LICENSE). Use it, fork it, teach with it.

<div align="center">

**Built one file at a time, on a laptop, until it made sense — and shared so it can make sense to you too.**
⭐ *Star it if it did.*

</div>
