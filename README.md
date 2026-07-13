# my-first-llm

A small GPT written from scratch in PyTorch, trained first on tiny-shakespeare
and later on TinyStories, as a learning project. The learning path and theory
live in [LERNPLAN.md](LERNPLAN.md) (German), the phase-by-phase implementation
plan in [AGENT_PLAN.md](AGENT_PLAN.md).

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
.venv/bin/python scripts/check_env.py
```

`check_env.py` must report `selected: mps` on the Mac. Mac-specific notes are
in [docs/SETUP.md](docs/SETUP.md).

## Phase 1: mini-GPT on tiny-shakespeare

```bash
# Download and tokenize the data (character level)
.venv/bin/python scripts/prepare_shakespeare.py

# Quick check that the training loop runs end to end (about a minute)
.venv/bin/python scripts/train.py --config configs/shakespeare_mini.py --smoke-test

# Full runs
.venv/bin/python scripts/train.py --config configs/shakespeare_bigram.py
.venv/bin/python scripts/train.py --config configs/shakespeare_mini.py

# Generate text from a checkpoint
.venv/bin/python scripts/generate.py --checkpoint checkpoints/shakespeare_mini/ckpt.pt \
    --prompt "ROMEO:" --max-new-tokens 500
```

Details and measured losses: [docs/PHASE1.md](docs/PHASE1.md).
