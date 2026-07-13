# Phase 1: bigram and mini-GPT on tiny-shakespeare

Goal of this phase: build a GPT from scratch and prove that the whole pipeline
(data, training loop, checkpointing, generation) works, on a dataset small
enough that a full run takes minutes rather than days.

## What was built

| File | Purpose |
|---|---|
| `scripts/prepare_shakespeare.py` | Downloads tiny-shakespeare, builds the character vocabulary, writes `train.bin`, `val.bin` (90/10 split, uint16) and `meta.json` |
| `src/training/data.py` | `CharTokenizer`, memory-mapped split loading, random batch sampling |
| `src/training/device.py` | Device selection (MPS, CUDA, CPU) shared by all scripts |
| `src/model/bigram.py` | Bigram baseline: a single `vocab x vocab` lookup table |
| `src/model/gpt.py` | GPT-2 style transformer: `CausalSelfAttention`, `MLP`, `Block`, `GPT` |
| `src/model/generate.py` | Autoregressive sampling with temperature and top-k |
| `src/training/loop.py` | AdamW, warmup plus cosine decay, gradient clipping and accumulation, validation loss, CSV log, checkpoint and resume |
| `scripts/train.py` | Entry point: `--config`, `--device`, `--smoke-test`, `--resume` |
| `scripts/generate.py` | Loads a checkpoint and samples text |

The two models share one interface: `forward(idx, targets=None)` returns
`(logits, loss)`. That is what lets the same training loop and the same
sampling function drive both.

## How to run

```bash
.venv/bin/python scripts/prepare_shakespeare.py

# Always validate the pipeline first (about 6 seconds)
.venv/bin/python scripts/train.py --config configs/shakespeare_mini.py --smoke-test

.venv/bin/python scripts/train.py --config configs/shakespeare_bigram.py
.venv/bin/python scripts/train.py --config configs/shakespeare_mini.py

.venv/bin/python scripts/generate.py \
    --checkpoint checkpoints/shakespeare_mini/ckpt.pt \
    --prompt "ROMEO:" --max-new-tokens 500
```

An interrupted run continues with
`--resume checkpoints/shakespeare_mini/ckpt.pt`.

## Data

Character level, no tokenizer yet (that is phase 2).

- 1,115,394 characters, vocabulary of 65 distinct characters
- 1,003,854 train tokens, 111,540 validation tokens

## Results

Measured on the MacBook Pro M4 (MPS, float32).

| Run | Parameters | Steps | Train loss | Val loss | Time |
|---|---|---|---|---|---|
| Bigram | 4,225 | 3,000 | 2.452 | 2.495 | 12 s |
| Mini-GPT | TBD | 5,000 | TBD | TBD | TBD |

Reference points for reading these numbers: random guessing over 65 characters
gives a loss of `ln(65) = 4.17` (the smoke test starts there, which is a good
sign that initialization is sane). The bigram model gets to 2.50 because it
knows the letter frequencies following any single character, and nothing more.
Everything below that is context the transformer actually uses.

### Sample output

TBD

## Observations

TBD
