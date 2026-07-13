"""Device selection shared by all scripts."""

import torch


def select_device(requested: str = "auto") -> torch.device:
    """Resolve the compute device to use.

    :param requested: One of "auto", "mps", "cuda", "cpu". With "auto" the best
        available backend is chosen (MPS, then CUDA, then CPU).
    :return: The resolved torch device.
    :raises ValueError: If an explicitly requested backend is unavailable.
    """
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")

    return torch.device(requested)
