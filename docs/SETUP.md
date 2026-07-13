# Setup and Mac-specific notes

## Environment

Python 3.11 via uv. The system Python 3.9 on macOS is too old and should not be
used.

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
```

Every script is run with the interpreter from the virtual environment, for
example `.venv/bin/python scripts/check_env.py`, so no manual `activate` is
needed.

## Apple Silicon (MPS)

- Device selection lives in `src/training/device.py`. All scripts take a
  `--device` flag that defaults to auto-detection: MPS, then CUDA, then CPU.
- Use `float32` on MPS. `bfloat16` and `torch.compile` are not reliable on this
  backend; speed is controlled through model and batch size instead.
- Some operators are not implemented for MPS. If a run fails with a missing
  operator, set `PYTORCH_ENABLE_MPS_FALLBACK=1` to fall back to CPU for those
  ops (this is slow, so it is a debugging aid, not a fix).
- CPU and GPU share the 16 GB of unified memory. Model, optimizer state, and
  batches must all fit at once; roughly 10 to 12 GB are realistically usable.

## Long training runs

- Run on the power supply, with good ventilation, and disable sleep. Thermal
  throttling and sleep are the two most common ways a long run dies.
- Every training run writes checkpoints and can be continued with
  `--resume <path>`, so an interrupted run is not lost.

## Formatting

Python code is formatted with black:

```bash
.venv/bin/black src scripts
.venv/bin/black --check src scripts
```
