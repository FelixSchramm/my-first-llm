# Agent Implementation Plan: my-first-llm

Step-by-step implementation plan for a coding agent. The learning rationale,
theory, and sources live in [LERNPLAN.md](LERNPLAN.md) (German); read it
before starting. This plan is self-contained enough to execute phase by
phase.

## Ground rules for the agent

- **Coding standards (mandatory):** Python code in English, PEP 8, formatted
  with `black`, `snake_case` names, no emojis anywhere. Every function gets a
  reStructuredText docstring (`:param name:` / `:return:`). Inline comments
  explain WHY, not WHAT. Every module/script gets a short accompanying
  Markdown file in `docs/`.
- **Do it simply.** This is a learning project: prefer plain, readable
  PyTorch over clever abstractions. No Hugging Face `Trainer`, no Lightning.
  External libraries allowed: `torch`, `numpy`, `datasets` (download only),
  `tiktoken` (comparison only), `gradio`, `fastapi`, `mlx-lm` (phase 6).
- **Hardware split:** This repo may be edited on Windows, but all training
  runs happen on the user's MacBook Pro M4 (16 GB unified memory) using the
  PyTorch MPS backend. Every training script must select its device via a
  `--device` flag with auto-detect default (`mps` if available, else `cuda`,
  else `cpu`) and must not hard-code paths.
- **Smoke test first.** Every training script needs a `--smoke-test` flag
  (tiny model, ~100 steps) and each phase's run must be validated in smoke
  mode before a long run is started.
- **Checkpoint and resume.** Long-running training must save checkpoints
  (model, optimizer, step, config, tokenizer reference) every N steps and
  support `--resume <checkpoint>`.
- **No git operations.** The user explicitly asked for no repository
  initialization, branches, or commits unless they request it later.
- **Do not commit/store large files carelessly:** datasets go to `data/`,
  checkpoints to `checkpoints/`; both must stay out of any future VCS.

## Target repository layout

```
my-first-llm/
  docs/                  # LERNPLAN.md, AGENT_PLAN.md, one MD per module
  src/
    tokenizer/           # phase 2: BPE implementation
    model/               # phase 1+3: GPT modules (attention, block, gpt)
    training/            # train loop, lr schedule, checkpointing, data loader
    inference/           # phase 4: sampling, kv-cache, cli, web ui
    sft/                 # phase 5: instruction formatting + SFT loop
  scripts/               # entry points (prepare_data.py, train.py, ...)
  configs/               # small python/json config files per experiment
  data/                  # downloaded + tokenized data (not versioned)
  checkpoints/           # model checkpoints (not versioned)
  notebooks/             # optional exploration
  requirements.txt
  README.md
```

---

## Phase 0 — Environment and scaffolding

- [ ] Create the directory layout above with placeholder `__init__.py` files.
- [ ] `requirements.txt`: torch, numpy, datasets, tiktoken, gradio, black.
- [ ] `scripts/check_env.py`: prints Python version, torch version, and
      device availability (`torch.backends.mps.is_available()`, CUDA, CPU);
      runs one small matmul on the selected device.
- [ ] `README.md`: one-paragraph project description, pointers to both plan
      documents, setup instructions (`python -m venv .venv`, pip install).
- [ ] `docs/SETUP.md`: Mac-specific notes (MPS fallback env var
      `PYTORCH_ENABLE_MPS_FALLBACK=1`, float32 on MPS, run on power supply).

**Acceptance:** `python scripts/check_env.py` passes on the Mac and reports
MPS as the selected device. `black --check src scripts` passes.

## Phase 1 — Micro foundation: bigram and mini-GPT on tiny-shakespeare

- [ ] `scripts/prepare_shakespeare.py`: download tiny-shakespeare (raw URL
      from the nanoGPT repo), build char-level vocab, save train/val token
      arrays (90/10 split) as uint16 `.bin` plus a `meta.json` (vocab).
- [ ] `src/model/bigram.py`: embedding-only bigram LM (Karpathy style).
- [ ] `src/model/gpt.py`: GPT-2-style decoder-only transformer, built from
      scratch: `CausalSelfAttention`, `MLP` (GELU), `Block` (pre-norm
      LayerNorm + residuals), learned positional embeddings, LM head.
      Config as a small dataclass (n_layer, n_head, d_model, block_size,
      vocab_size, dropout).
- [ ] `src/training/loop.py`: generic training loop with AdamW, linear
      warmup + cosine decay, gradient clipping (1.0), gradient accumulation,
      periodic val-loss estimation, CSV loss logging
      (`checkpoints/<run>/log.csv`), checkpoint save/resume.
- [ ] `scripts/train.py`: entry point wiring config + data + loop; flags:
      `--config`, `--device`, `--smoke-test`, `--resume`.
- [ ] `scripts/generate.py`: load checkpoint, generate N tokens from a
      prompt (greedy + temperature sampling for now).
- [ ] `docs/PHASE1.md`: what was built, how to run, observed losses.

**Acceptance:**
- Bigram model reaches val loss ~2.5 on shakespeare (char-level).
- Mini-GPT (approx. 6 layers, d_model 384, block_size 256, ~10M params)
  reaches val loss < 1.6 and generates recognizably Shakespeare-like text.
- Training run of the mini config completes on MPS in well under an hour;
  interruption + `--resume` continues cleanly.

## Phase 2 — BPE tokenizer from scratch

- [ ] `scripts/prepare_tinystories.py`: download TinyStoriesV2 from
      Hugging Face (`roneneldan/TinyStories`, V2 files), store raw text in
      `data/tinystories/`.
- [ ] `src/tokenizer/bpe.py`: byte-level BPE trainer + encoder/decoder in
      the spirit of karpathy/minbpe (train, encode, decode, save/load as
      JSON). Include a special `<|endoftext|>` token as document separator.
- [ ] `scripts/train_tokenizer.py`: train on a TinyStories sample
      (e.g. 50-100 MB); default vocab size 8192 (config flag).
- [ ] Comparison script or notebook: tokens-per-character and example
      segmentations of our tokenizer vs. `tiktoken` GPT-2 on the same
      sample; results table into `docs/PHASE2.md`.
- [ ] `scripts/tokenize_tinystories.py`: encode the full corpus into
      train/val uint16 memmap `.bin` files with `<|endoftext|>` separators.

**Acceptance:** round-trip `decode(encode(s)) == s` on random samples;
compression ratio around 3.5-4.5 chars/token on TinyStories; full corpus
tokenizes without memory errors (stream in chunks).

## Phase 3 — Pretraining on TinyStories + architecture ablations

- [ ] `configs/tinystories_25m.py` (approx.: n_layer 8, n_head 8,
      d_model 512, block_size 512, vocab 8192 -> ~25M params) and a smaller
      `tinystories_10m.py` fallback if MPS throughput disappoints.
- [ ] Reuse the phase-1 training loop; add weight tying option
      (share token embedding and LM head).
- [ ] During training, sample 2-3 stories at every eval step into the run
      log so quality regressions are visible early.
- [ ] Long run on the Mac (expect several hours to ~1-2 days; document a
      Google Colab T4 fallback path in `docs/PHASE3.md`).
- [ ] Ablations (each as its own config + short run of fixed token budget,
      e.g. 100M tokens, same seed): (1) RoPE instead of learned positions,
      (2) RMSNorm instead of LayerNorm, (3) SwiGLU instead of GELU MLP,
      (4) no biases. Implement each behind a config switch in
      `src/model/gpt.py`. Record final val losses in a table in
      `docs/PHASE3.md`.

**Acceptance:** main model reaches val loss in the ~1.2-1.8 range and
generates multi-paragraph, grammatical, mostly coherent children's stories
from a 2-3 word prompt; ablation table complete with a short written
interpretation.

## Phase 4 — Inference: sampling, KV cache, CLI, web UI

- [ ] `src/inference/sampler.py`: temperature, top-k, top-p sampling over
      logits (composable, unit-tested with a fixed dummy distribution).
- [ ] `src/inference/engine.py`: generation with KV cache (store K/V per
      layer, feed only the new token per step); verify identical outputs
      with and without cache at temperature 0; measure tokens/sec speedup.
- [ ] `scripts/chat_cli.py`: REPL that streams tokens to the terminal.
- [ ] `scripts/serve_ui.py`: Gradio `ChatInterface` (story-completion mode:
      user prompt = story beginning); optional FastAPI `POST /generate`
      endpoint in `scripts/serve_api.py`.
- [ ] `docs/PHASE4.md` with usage and the measured KV-cache speedup.

**Acceptance:** cached and uncached generation are token-identical at
temperature 0; KV cache gives a clear speedup (expect >2x for 500-token
generations); Gradio UI works in the browser on the Mac.

## Phase 5 — SFT of the own model (instruction following, story style)

- [ ] Build an instruction dataset: template
      `<|user|>Write a story about {topic}.<|assistant|>{story}<|endoftext|>`
      where topics are extracted or paraphrased from TinyStories samples
      (few tens of thousands of examples suffice). Add the special tokens to
      the tokenizer.
- [ ] `src/sft/dataset.py` + `src/sft/train_sft.py`: fine-tune the phase-3
      checkpoint; compute loss only on assistant tokens (prompt masking).
- [ ] Before/after comparison in `docs/PHASE5.md`: same 10 prompts through
      base model and SFT model.

**Acceptance:** the SFT model reliably responds to "Write a story about X"
with a story about X (base model typically just continues text instead).
Expectations are calibrated: no general chat ability at 25M params.

## Phase 6 — Fine-tune a real model with MLX and host it with Ollama

All steps run on the Mac.

- [ ] Install `mlx-lm`; pick base model: Qwen3-4B (or Qwen2.5-3B /
      Llama-3.2-3B) in 4-bit from the mlx-community Hugging Face org.
- [ ] Choose/prepare an SFT dataset: either a small public one (Alpaca
      subset, SmolTalk subset) or a personal one (e.g. style samples);
      convert to the mlx-lm `train.jsonl` chat format.
- [ ] Run QLoRA training (`mlx_lm.lora --train ...`), monitor val loss,
      keep run small first (few hundred iterations) before a longer run.
- [ ] Evaluate: `mlx_lm.generate` with adapter vs. without on a fixed
      prompt set.
- [ ] Fuse adapters (`mlx_lm.fuse`), convert with llama.cpp
      `convert_hf_to_gguf.py` (outtype q4_K_M or f16 + `llama-quantize`),
      write an Ollama `Modelfile` (FROM ./model.gguf + chat template +
      params), `ollama create my-first-llm`, test with `ollama run`.
- [ ] `docs/PHASE6.md`: full command log, memory observations, before/after
      examples.

**Acceptance:** `ollama run my-first-llm` answers in the fine-tuned style;
peak memory stayed within 16 GB during training (log it); the whole path
from base model to hosted GGUF is reproducible from PHASE6.md alone.

---

## Verification summary (per phase)

| Phase | Verify by |
|---|---|
| 0 | `check_env.py` on Mac selects MPS; black passes |
| 1 | val-loss targets hit; resume works; samples readable |
| 2 | round-trip lossless; compression ratio plausible |
| 3 | coherent stories; ablation table with fixed token budget |
| 4 | cache/no-cache identical at temp 0; speedup measured |
| 5 | 10-prompt before/after shows instruction following |
| 6 | model runs in Ollama; commands reproducible |
