import optax
from absl import logging
from flax.training.train_state import TrainState

from flax import nnx


def get_optimizer(config, model):
    if config.params.do_warmup and config.params.do_decay:
        lr = optax.warmup_cosine_decay_schedule(
            init_value=config.params.init_learning_rate,
            peak_value=config.params.learning_rate,
            warmup_steps=config.params.warmup_steps,
            decay_steps=config.params.decay_steps,
            end_value=config.params.end_learning_rate,
        )
    elif config.params.do_warmup:
        lr = optax.linear_schedule(
            init_value=config.params.init_learning_rate,
            end_value=config.params.learning_rate,
            transition_steps=config.params.warmup_steps,
        )
    elif config.params.do_decay:
        lr = optax.cosine_decay_schedule(
            init_value=config.params.learning_rate,
            decay_steps=config.params.decay_steps,
            alpha=config.params.end_learning_rate / config.params.learning_rate,
        )
    else:
        lr = config.params.learning_rate

    if config.name == "adamw":
        tx = optax.adamw(
            lr,
            b1=config.params.b1,
            b2=config.params.b2,
            weight_decay=config.params.weight_decay,
        )
    elif config.name == "radam":
        tx = optax.radam(lr)
    else:
        tx = optax.adam(lr, b1=config.params.b1, b2=config.params.b2)

    if config.params.do_gradient_clipping:
        tx = optax.chain(
            optax.clip_by_global_norm(config.params.gradient_clipping), tx
        )

    tx = nnx.Optimizer(
        model,
        tx=tx,
    )
    return tx
