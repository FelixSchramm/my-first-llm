"""Phase 0 environment check: report versions, devices, and run a test matmul."""

import argparse
import platform
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.device import select_device


def report_environment() -> None:
    """Print Python, torch and backend availability information.

    :return: None.
    """
    print(f"python:        {platform.python_version()}")
    print(f"platform:      {platform.platform()}")
    print(f"torch:         {torch.__version__}")
    print(f"mps available: {torch.backends.mps.is_available()}")
    print(f"mps built:     {torch.backends.mps.is_built()}")
    print(f"cuda available:{torch.cuda.is_available()}")


def run_test_matmul(device: torch.device) -> None:
    """Run a small matrix multiplication to prove the device really works.

    :param device: The device to run the multiplication on.
    :return: None.
    """
    a = torch.randn(512, 512, device=device)
    b = torch.randn(512, 512, device=device)
    c = a @ b
    # Force execution: MPS and CUDA queue work asynchronously, so without
    # reading a value back the matmul might not have run at all.
    checksum = c.sum().item()
    print(f"matmul on {device}: ok (checksum {checksum:.2f})")


def main() -> None:
    """Parse arguments, report the environment, and test the selected device.

    :return: None.
    """
    parser = argparse.ArgumentParser(description="Check the training environment")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Device to test (default: auto-detect)",
    )
    args = parser.parse_args()

    report_environment()
    device = select_device(args.device)
    print(f"selected:      {device}")
    run_test_matmul(device)


if __name__ == "__main__":
    main()
