import optax
import orbax.checkpoint as ocp
from absl import logging
from flax import nnx


def get_optimizer(config, model, which):
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
        tx = optax.adamw(lr, weight_decay=config.params.weight_decay)
    elif config.name == "radam":
        tx = optax.radam(lr)
    else:
        tx = optax.adam(lr)

    if config.params.do_gradient_clipping:
        tx = optax.chain(
            optax.clip_by_global_norm(config.params.gradient_clipping), tx
        )

    tx = nnx.Optimizer(
        model,
        tx=tx,
    )
    return tx


def get_checkpointer_fns(outfolder, config):
    options = ocp.CheckpointManagerOptions(
        max_to_keep=config.max_to_keep,
        save_interval_steps=config.save_interval_steps,
        create=True,
    )
    checkpointer = ocp.PyTreeCheckpointer()
    checkpoint_manager = ocp.CheckpointManager(
        outfolder,
        checkpointer,
        options,
    )

    def save_fn(step, model_fns, metrics):
        state = nnx.state(model_fns)
        checkpoint_manager.save(
            step=step,
            items=state,
            metrics=metrics,
        )
        checkpoint_manager.wait_until_finished()

    return checkpoint_manager, save_fn


def get_latest_train_state(mngr, model_fns):
    try:
        logging.info("trying to load train state")
        model_fns = _restore_train_state(mngr, model_fns, "latest")
        logging.info("successfully restored train state")
        return model_fns, mngr.latest_step()
    except Exception as e:
        logging.info(str(e))
        logging.info("training from scratch")
        pass
    return model_fns, 0


def _restore_train_state(mngr, model_fns, which):
    step = mngr.latest_step() if which == "latest" else mngr.best_step()
    model_fns = nnx.eval_shape(lambda: model_fns)
    state = mngr.restore(step, item=model_fns)
    nnx.update(model_fns, state)
    return model_fns
