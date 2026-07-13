"""Entry point for training: load a config, build the model, run the loop."""

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import build_model
from src.training.data import CharTokenizer, load_split
from src.training.device import select_device
from src.training.loop import train


def load_config(path: Path) -> ModuleType:
    """Import a config file by path.

    :param path: Path to the config module.
    :return: The imported module, exposing model_type, model, and train.
    """
    spec = importlib.util.spec_from_file_location("run_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_smoke_test(config: ModuleType) -> None:
    """Shrink a config so a full run finishes in about a minute.

    The point is to prove that data loading, training, evaluation, logging, and
    checkpointing all work before a long run is started.

    :param config: The loaded config module, modified in place.
    :return: None.
    """
    model, train_config = config.model, config.train

    model.block_size = min(model.block_size, 64)
    if config.model_type == "gpt":
        model.n_layer = 2
        model.n_head = 2
        model.d_model = 64

    train_config.run_name = f"{train_config.run_name}_smoke"
    train_config.batch_size = 8
    train_config.max_steps = 100
    train_config.warmup_steps = 10
    train_config.eval_interval = 50
    train_config.eval_iters = 20
    train_config.checkpoint_interval = 50


def main() -> None:
    """Parse arguments, wire config, data, and model together, and train.

    :return: None.
    """
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", required=True, help="Path to a config file")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Device to train on (default: auto-detect)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny model for 100 steps to validate the pipeline",
    )
    parser.add_argument("--resume", help="Checkpoint to continue from")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if args.smoke_test:
        apply_smoke_test(config)

    data_dir = Path(config.train.data_dir)
    tokenizer = CharTokenizer.from_meta(data_dir)
    # The vocabulary comes from the data, never from the config, so model and
    # data can never disagree about it.
    config.model.vocab_size = tokenizer.vocab_size

    splits = {split: load_split(data_dir, split) for split in ("train", "val")}
    device = select_device(args.device)
    model = build_model(config.model_type, config.model)

    print(f"run:        {config.train.run_name}")
    print(f"device:     {device}")
    print(f"vocab size: {tokenizer.vocab_size}")
    print(f"parameters: {model.num_parameters():,}")

    train(
        model=model,
        model_type=config.model_type,
        model_config=config.model,
        config=config.train,
        splits=splits,
        device=device,
        resume_from=Path(args.resume) if args.resume else None,
    )


if __name__ == "__main__":
    main()
