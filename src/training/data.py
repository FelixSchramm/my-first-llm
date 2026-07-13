"""Character-level data loading from the uint16 token arrays on disk."""

import json
from pathlib import Path

import numpy as np
import torch


class CharTokenizer:
    """Character-level encoder and decoder built from a stored vocabulary."""

    def __init__(self, itos: list[str]) -> None:
        """Create the tokenizer from an index-to-character table.

        :param itos: Characters in vocabulary order.
        :return: None.
        """
        self.itos = itos
        self.stoi = {ch: i for i, ch in enumerate(itos)}
        self.vocab_size = len(itos)

    @classmethod
    def from_meta(cls, data_dir: Path) -> "CharTokenizer":
        """Load the tokenizer written by ``scripts/prepare_shakespeare.py``.

        :param data_dir: Directory containing ``meta.json``.
        :return: The tokenizer described by that file.
        """
        meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
        return cls(meta["itos"])

    def encode(self, text: str) -> list[int]:
        """Encode a string into token ids.

        :param text: The string to encode.
        :return: The token ids.
        """
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back into a string.

        :param ids: The token ids to decode.
        :return: The decoded string.
        """
        return "".join(self.itos[i] for i in ids)


def load_split(data_dir: Path, split: str) -> np.ndarray:
    """Memory-map one token array from disk.

    :param data_dir: Directory containing ``train.bin`` and ``val.bin``.
    :param split: Either "train" or "val".
    :return: The tokens as a read-only uint16 array.
    """
    return np.memmap(data_dir / f"{split}.bin", dtype=np.uint16, mode="r")


def get_batch(
    tokens: np.ndarray,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of input and target sequences.

    Targets are the inputs shifted by one position, which is exactly the
    next-token prediction task.

    :param tokens: The token array to sample from.
    :param batch_size: Number of sequences in the batch.
    :param block_size: Length of each sequence (the context length).
    :param device: Device the batch is moved to.
    :return: Tuple of input and target tensors, each of shape (batch, block).
    """
    starts = torch.randint(len(tokens) - block_size - 1, (batch_size,))
    # int64 because the embedding lookup and cross entropy both need long tensors.
    x = torch.stack(
        [torch.from_numpy(tokens[i : i + block_size].astype(np.int64)) for i in starts]
    )
    y = torch.stack(
        [
            torch.from_numpy(tokens[i + 1 : i + 1 + block_size].astype(np.int64))
            for i in starts
        ]
    )
    return x.to(device), y.to(device)
