"""Autoregressive sampling shared by the bigram model and the GPT.

This is the simple version for phase 1: no KV cache, the whole context is fed
through the model again for every new token. Phase 4 replaces it.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


@torch.no_grad()
def generate(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Extend a token sequence by sampling from the model.

    :param model: A model whose forward returns (logits, loss).
    :param idx: Context token ids of shape (batch, time).
    :param max_new_tokens: Number of tokens to append.
    :param temperature: Softmax temperature. Values below 1.0 make the output
        more conservative, above 1.0 more random. Must be positive.
    :param top_k: If given, sample only from the k most likely tokens.
    :return: The context with the generated tokens appended.
    :raises ValueError: If the temperature is not positive.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    model.eval()
    block_size = model.config.block_size
    for _ in range(max_new_tokens):
        # The model has no memory, so the context is capped at the block size.
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        # Only the last position predicts the next token.
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            k = min(top_k, logits.size(-1))
            threshold = torch.topk(logits, k, dim=-1).values[:, [-1]]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)

    return idx
