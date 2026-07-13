# my-first-llm - AI Coding Instructions

Learning project: build a GPT from scratch in PyTorch, then fine-tune and host
an open-source model. The learning path is in [LERNPLAN.md](LERNPLAN.md), the
phase-by-phase implementation plan in [AGENT_PLAN.md](AGENT_PLAN.md). Read both
before starting work.

**This is a learning project.** The value is in understanding every component,
not in shipping fast. Prefer plain, readable code over clever code, and explain
the reasoning where it is not obvious from the code itself.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes,
simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Every phase in AGENT_PLAN.md has acceptance criteria. Use them: a phase is done
when its criteria are demonstrably met, not when the code exists.

For multi-step tasks, state a brief plan:
```
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
```

## Coding standards

- Write code in English, as simple as possible, and do not use emojis.
- Follow PEP 8. Variable and function names in snake_case.
- Format Python with `black`: `.venv/bin/black src scripts configs`.
- Use reStructuredText (reST) docstrings: `:param name: description` and
  `:return: description`. Every function gets one.
- Inline comments explain WHY, not WHAT.
- Allowed libraries: `torch`, `numpy`, `datasets` (download only), `tiktoken`
  (comparison only), `gradio`, `fastapi`, `mlx-lm` (phase 6). No Hugging Face
  `Trainer`, no Lightning - the point is to write the loop yourself.

## Environment

- Python 3.11, managed with `uv`. Run scripts with the interpreter from the
  virtual environment: `.venv/bin/python scripts/train.py ...`.
- Training runs on a MacBook Pro M4 (16 GB unified memory) via the PyTorch MPS
  backend. Every script selects its device through `src/training/device.py`
  with a `--device` flag that auto-detects (MPS, then CUDA, then CPU).
- On MPS use `float32`. `bfloat16` and `torch.compile` are not reliable there.
- Details and pitfalls: [docs/SETUP.md](docs/SETUP.md).

## Training runs

- **Smoke test first.** Every training script has a `--smoke-test` flag (tiny
  model, 100 steps). Run it before starting a long run. The most expensive bug
  is the one found twelve hours into a run.
- **Checkpoint and resume.** Long runs must save checkpoints and support
  `--resume <checkpoint>`. A laptop sleeps, updates, and runs out of battery.
- **Log the loss and read the samples.** A falling loss with degenerate output
  means the data pipeline is broken, and only the samples show that.
- Model and data must always agree on the vocabulary: the vocab size comes from
  the data, never from a config file.

## Documentation

- Every phase gets a Markdown file in `docs/` (`docs/PHASE1.md`, ...) with what
  was built, how to run it, and the measured results.
- Record actual numbers (losses, timings, parameter counts), not intentions.

## Git

- Conventional commits (https://www.conventionalcommits.org/).
- Work on feature branches, e.g. `feat/phase-2-bpe-tokenizer`. This is a
  private learning repo, so there are no Jira ticket numbers in branch names.
- Pull requests describe: what was built, the measured results against the
  phase's acceptance criteria, and how it was verified.
- `data/` and `checkpoints/` are gitignored and must stay out of the repo.
