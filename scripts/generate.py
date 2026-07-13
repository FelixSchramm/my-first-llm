"""Load a checkpoint and generate text from a prompt."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import build_config, build_model
from src.model.generate import generate
from src.training.data import CharTokenizer
from src.training.device import select_device


def main() -> None:
    """Parse arguments, restore the model, and print a sample.

    :return: None.
    """
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to ckpt.pt")
    parser.add_argument("--prompt", default="\n", help="Text to continue")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Below 1.0 is more conservative, above 1.0 more random",
    )
    parser.add_argument("--top-k", type=int, help="Sample only from the k best tokens")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Device to generate on (default: auto-detect)",
    )
    args = parser.parse_args()

    device = select_device(args.device)
    torch.manual_seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = build_config(checkpoint["model_type"], checkpoint["model_config"])
    model = build_model(checkpoint["model_type"], model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    # The checkpoint records the data it was trained on, which is where the
    # vocabulary lives.
    tokenizer = CharTokenizer.from_meta(Path(checkpoint["data_dir"]))

    context = torch.tensor(
        [tokenizer.encode(args.prompt)], dtype=torch.long, device=device
    )
    out = generate(
        model,
        context,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
