"""Training loop: AdamW, warmup plus cosine decay, checkpointing, CSV logging."""

import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.training.data import get_batch


@dataclass
class TrainConfig:
    """Configuration of a training run.

    :param run_name: Name of the run; also the checkpoint directory name.
    :param data_dir: Directory holding train.bin, val.bin and meta.json.
    :param batch_size: Sequences per micro-batch.
    :param grad_accum_steps: Micro-batches summed before one optimizer step.
    :param max_steps: Number of optimizer steps.
    :param learning_rate: Peak learning rate after warmup.
    :param min_lr_ratio: Final learning rate as a fraction of the peak.
    :param warmup_steps: Steps of linear warmup at the start.
    :param weight_decay: AdamW weight decay, applied to matrices only.
    :param grad_clip: Maximum gradient norm; 0 disables clipping.
    :param eval_interval: Steps between validation-loss estimates.
    :param eval_iters: Batches averaged per loss estimate.
    :param checkpoint_interval: Steps between checkpoint writes.
    :param seed: Random seed.
    """

    run_name: str
    data_dir: str
    batch_size: int = 64
    grad_accum_steps: int = 1
    max_steps: int = 5000
    learning_rate: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 100
    checkpoint_interval: int = 500
    seed: int = 1337


def get_lr(step: int, config: TrainConfig) -> float:
    """Compute the learning rate for a step: linear warmup, then cosine decay.

    Without warmup transformers tend to diverge in the first few hundred steps,
    because the Adam moment estimates are still unreliable.

    :param step: The current optimizer step, starting at 0.
    :param config: The training configuration.
    :return: The learning rate to use for this step.
    """
    min_lr = config.learning_rate * config.min_lr_ratio

    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps

    decay_steps = max(1, config.max_steps - config.warmup_steps)
    progress = min(1.0, (step - config.warmup_steps) / decay_steps)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coefficient * (config.learning_rate - min_lr)


def create_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.AdamW:
    """Build the AdamW optimizer with weight decay on matrices only.

    Biases and norm parameters are one-dimensional and are excluded from weight
    decay, which is the standard GPT-2 recipe.

    :param model: The model to optimize.
    :param config: The training configuration.
    :return: The configured optimizer.
    """
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)

    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=config.learning_rate, betas=(0.9, 0.99))


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    splits: dict[str, np.ndarray],
    config: TrainConfig,
    block_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Average the loss over several batches of each split.

    A single batch is far too noisy to judge progress, hence the average.

    :param model: The model to evaluate.
    :param splits: Mapping of split name to token array.
    :param config: The training configuration.
    :param block_size: Context length of the model.
    :param device: Device to run on.
    :return: Mapping of split name to mean loss.
    """
    model.eval()
    losses = {}
    for split, tokens in splits.items():
        values = torch.zeros(config.eval_iters)
        for i in range(config.eval_iters):
            x, y = get_batch(tokens, config.batch_size, block_size, device)
            _, loss = model(x, y)
            values[i] = loss.item()
        losses[split] = values.mean().item()
    model.train()
    return losses


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    model_type: str,
    model_config: object,
    train_config: TrainConfig,
) -> None:
    """Write a checkpoint that is sufficient to resume or to generate text.

    :param path: File to write.
    :param model: The model whose weights are stored.
    :param optimizer: The optimizer whose state is stored.
    :param step: The number of completed optimizer steps.
    :param model_type: Either "bigram" or "gpt".
    :param model_config: The model configuration dataclass.
    :param train_config: The training configuration.
    :return: None.
    """
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "model_type": model_type,
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            # The tokenizer lives with the data, so the model can never be
            # loaded with a vocabulary it was not trained on.
            "data_dir": train_config.data_dir,
        },
        path,
    )


def train(
    model: nn.Module,
    model_type: str,
    model_config: object,
    config: TrainConfig,
    splits: dict[str, np.ndarray],
    device: torch.device,
    resume_from: Path | None = None,
) -> None:
    """Run the training loop and write checkpoints and a CSV loss log.

    :param model: The model to train.
    :param model_type: Either "bigram" or "gpt".
    :param model_config: The model configuration dataclass.
    :param config: The training configuration.
    :param splits: Mapping of "train" and "val" to their token arrays.
    :param device: Device to train on.
    :param resume_from: Checkpoint to continue from, or None to start fresh.
    :return: None.
    """
    torch.manual_seed(config.seed)
    block_size = model_config.block_size

    run_dir = Path("checkpoints") / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "ckpt.pt"
    log_path = run_dir / "log.csv"

    model.to(device)
    optimizer = create_optimizer(model, config)

    start_step = 0
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"]
        print(f"resumed from {resume_from} at step {start_step}")

    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(["step", "train_loss", "val_loss", "lr"])

    model.train()
    start_time = time.time()

    for step in range(start_step, config.max_steps):
        lr = get_lr(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        for _ in range(config.grad_accum_steps):
            x, y = get_batch(splits["train"], config.batch_size, block_size, device)
            _, loss = model(x, y)
            # Scale so that the accumulated gradient equals the gradient of the
            # mean loss over the full effective batch.
            (loss / config.grad_accum_steps).backward()

        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        is_last = step == config.max_steps - 1
        if step % config.eval_interval == 0 or is_last:
            losses = estimate_loss(model, splits, config, block_size, device)
            elapsed = time.time() - start_time
            print(
                f"step {step:5d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | lr {lr:.2e} | {elapsed:.0f}s"
            )
            with log_path.open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(
                    [
                        step,
                        f"{losses['train']:.4f}",
                        f"{losses['val']:.4f}",
                        f"{lr:.6f}",
                    ]
                )

        if (step + 1) % config.checkpoint_interval == 0 or is_last:
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                step + 1,
                model_type,
                model_config,
                config,
            )

    print(f"done in {time.time() - start_time:.0f}s, checkpoint at {checkpoint_path}")
