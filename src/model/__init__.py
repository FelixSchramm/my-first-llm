"""Model package: bigram baseline and GPT, plus a factory used by the scripts."""

import torch.nn as nn

from src.model.bigram import BigramConfig, BigramLanguageModel
from src.model.gpt import GPT, GPTConfig


def build_model(model_type: str, config: object) -> nn.Module:
    """Instantiate a model from its type name and configuration.

    :param model_type: Either "bigram" or "gpt".
    :param config: The matching configuration dataclass.
    :return: The instantiated model.
    :raises ValueError: If the model type is unknown.
    """
    if model_type == "bigram":
        return BigramLanguageModel(config)
    if model_type == "gpt":
        return GPT(config)
    raise ValueError(f"unknown model type: {model_type}")


def build_config(model_type: str, values: dict) -> object:
    """Rebuild a model configuration from a stored checkpoint dict.

    :param model_type: Either "bigram" or "gpt".
    :param values: The configuration fields as stored in the checkpoint.
    :return: The reconstructed configuration dataclass.
    :raises ValueError: If the model type is unknown.
    """
    if model_type == "bigram":
        return BigramConfig(**values)
    if model_type == "gpt":
        return GPTConfig(**values)
    raise ValueError(f"unknown model type: {model_type}")
