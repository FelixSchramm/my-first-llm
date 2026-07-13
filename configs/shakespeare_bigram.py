"""Bigram baseline on tiny-shakespeare. Target: val loss around 2.5."""

from src.model.bigram import BigramConfig
from src.training.loop import TrainConfig

model_type = "bigram"

model = BigramConfig(
    block_size=8,
)

train = TrainConfig(
    run_name="shakespeare_bigram",
    data_dir="data/shakespeare",
    batch_size=32,
    max_steps=3000,
    learning_rate=1e-2,
    warmup_steps=100,
    weight_decay=0.0,
    eval_interval=250,
    eval_iters=200,
    checkpoint_interval=1000,
)
