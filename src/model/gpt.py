"""GPT-2 style decoder-only transformer, built from scratch.

Structure: token and position embeddings, then N pre-norm blocks of causal
self-attention plus MLP, then a final norm and a linear head projecting to the
vocabulary.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """Configuration of the GPT model.

    :param block_size: Maximum context length in tokens.
    :param vocab_size: Number of tokens. Filled in from the data at train time.
    :param n_layer: Number of transformer blocks.
    :param n_head: Number of attention heads per block.
    :param d_model: Embedding dimension.
    :param dropout: Dropout probability.
    :param bias: Whether linear layers and norms use bias terms.
    """

    block_size: int = 256
    vocab_size: int = 0
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    dropout: float = 0.2
    bias: bool = True


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where a token can only attend to the past."""

    def __init__(self, config: GPTConfig) -> None:
        """Build the projections and the causal mask.

        :param config: The model configuration.
        :return: None.
        """
        super().__init__()
        if config.d_model % config.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

        self.n_head = config.n_head
        self.d_model = config.d_model
        self.dropout = config.dropout

        # Query, key and value for all heads in one matrix, split after the fact.
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Lower-triangular mask, registered as a buffer so it moves with the
        # model to the device but is not a trained parameter.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal multi-head attention.

        :param x: Input of shape (batch, time, d_model).
        :return: Output of shape (batch, time, d_model).
        """
        b, t, c = x.shape
        head_dim = c // self.n_head

        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        # (batch, head, time, head_dim) so that attention runs per head.
        q = q.view(b, t, self.n_head, head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, head_dim).transpose(1, 2)

        # Scaling by sqrt(head_dim) keeps the softmax from saturating as the
        # head dimension grows.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward network with a 4x hidden expansion."""

    def __init__(self, config: GPTConfig) -> None:
        """Build the two linear layers.

        :param config: The model configuration.
        :return: None.
        """
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network.

        :param x: Input of shape (batch, time, d_model).
        :return: Output of the same shape.
        """
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """One transformer block: pre-norm attention and MLP, both with residuals."""

    def __init__(self, config: GPTConfig) -> None:
        """Build the norms, the attention, and the MLP.

        :param config: The model configuration.
        :return: None.
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block.

        :param x: Input of shape (batch, time, d_model).
        :return: Output of the same shape.
        """
        # Pre-norm: normalize before the sublayer, add the raw input back. This
        # keeps a clean residual path from input to output and trains more
        # stably than the original post-norm ordering.
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """Decoder-only transformer language model."""

    def __init__(self, config: GPTConfig) -> None:
        """Build embeddings, blocks, final norm, and the language-model head.

        :param config: The model configuration.
        :return: None.
        """
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model, bias=config.bias)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        # Scale the residual projections down by the number of layers they feed
        # into, so the variance of the residual stream does not grow with depth
        # (GPT-2 paper, section 2.3).
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(
                    param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize linear and embedding weights the way GPT-2 does.

        :param module: The module to initialize.
        :return: None.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Map token ids to next-token logits and optionally the loss.

        :param idx: Token ids of shape (batch, time).
        :param targets: Target token ids of shape (batch, time), or None.
        :return: Tuple of logits (batch, time, vocab) and the cross-entropy loss
            (None if no targets were given).
        :raises ValueError: If the sequence is longer than the context window.
        """
        b, t = idx.shape
        if t > self.config.block_size:
            raise ValueError(
                f"sequence of length {t} exceeds block size "
                f"{self.config.block_size}"
            )

        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.token_embedding(idx) + self.position_embedding(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        b, t, v = logits.shape
        loss = F.cross_entropy(logits.view(b * t, v), targets.view(b * t))
        return logits, loss

    def num_parameters(self) -> int:
        """Count the trainable parameters.

        :return: The number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
