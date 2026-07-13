"""Bigram language model: the simplest possible baseline.

The model has no context at all. It looks up one row of a vocab x vocab table
per token, so its prediction depends only on the single previous token. It
exists to make the training loop and the data pipeline testable, and to give a
loss baseline that the transformer has to beat.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class BigramConfig:
    """Configuration of the bigram model.

    :param vocab_size: Number of tokens. Filled in from the data at train time.
    :param block_size: Sequence length used by the data loader. The model itself
        has no context window, but the loop needs a length to sample batches.
    """

    block_size: int = 8
    vocab_size: int = 0


class BigramLanguageModel(nn.Module):
    """Predicts the next token from a lookup table indexed by the current one."""

    def __init__(self, config: BigramConfig) -> None:
        """Build the token-to-logits lookup table.

        :param config: The model configuration.
        :return: None.
        """
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Map token ids to next-token logits and optionally the loss.

        :param idx: Token ids of shape (batch, time).
        :param targets: Target token ids of shape (batch, time), or None.
        :return: Tuple of logits (batch, time, vocab) and the cross-entropy loss
            (None if no targets were given).
        """
        logits = self.token_embedding(idx)
        if targets is None:
            return logits, None

        # Cross entropy expects (N, classes), so the batch and time axes are
        # flattened into one.
        b, t, v = logits.shape
        loss = F.cross_entropy(logits.view(b * t, v), targets.view(b * t))
        return logits, loss

    def num_parameters(self) -> int:
        """Count the trainable parameters.

        :return: The number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
