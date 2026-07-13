"""Mini-GPT on tiny-shakespeare, about 10M parameters.

Target: val loss below 1.6 and recognizably Shakespeare-like text. This is the
configuration from Karpathy's "Let's build GPT".
"""

from src.model.gpt import GPTConfig
from src.training.loop import TrainConfig

model_type = "gpt"

model = GPTConfig(
    block_size=256,
    n_layer=6,
    n_head=6,
    d_model=384,
    dropout=0.2,
)

train = TrainConfig(
    run_name="shakespeare_mini",
    data_dir="data/shakespeare",
    batch_size=64,
    max_steps=5000,
    learning_rate=1e-3,
    warmup_steps=100,
    eval_interval=250,
    eval_iters=100,
    checkpoint_interval=500,
)
