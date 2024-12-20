import hashlib
import os

import chex
import jax
import matplotlib.pyplot as plt
import numpy as np
import wandb
from absl import app, flags, logging
from flax import jax_utils
from flax.training import common_utils
from jax import numpy as jnp
from jax import random as jr
from jax.lib import xla_bridge
from ml_collections import config_flags

from checkpointer import new_train_state
from dataloader import data_loaders
from models import make_model
from wgan.loss import WGANGP, WGANGPConfig

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "model configuration")
flags.DEFINE_string("workdir", None, "work directory")
flags.DEFINE_bool("usewand", False, "use wandb for logging")
flags.mark_flags_as_required(["workdir", "config"])


def get_model_and_step_fn(config):
    generator, critic = make_model(
        shape=config.data.shape, config=config.nn
    )
    step_fn, eval_fn = WGANGP(generator.apply, critic.apply, WGANGPConfig(config.training.n_update_generator))
    return (generator, critic), (step_fn, eval_fn)


def metrics_to_summary(train_metrics, val_metrics):
    train_metrics = common_utils.get_metrics(train_metrics)
    val_metrics = common_utils.get_metrics(val_metrics)
    train_summary = {
        f"train/{k}": v
        for k, v in jax.tree.map(
            lambda x: float(x.mean()), train_metrics
        ).items()
    }
    val_summary = {
        f"val/{k}": v
        for k, v in jax.tree.map(lambda x: float(x.mean()), val_metrics).items()
    }
    return train_summary | val_summary


def init_critic(rng_key, model, config):
    params = model.init(
        {"params": rng_key, "sample": rng_key},
        inputs=jnp.ones((10, *config.shape)),
        context=jnp.arange(10),
        is_training=False,
    )
    return params


def init_generator(rng_key, model, config):
    params = model.init(
        {"params": rng_key, "sample": rng_key},
        sample_shape=(10,),
        context=None,
        is_training=False,
    )
    return params


def train(
    rng_key, models, matching_fns, config, train_iter, val_iter, model_id
):
    init_key, rng_key = jr.split(rng_key)    
    
    step_fn, eval_fn = matching_fns
    generator_fn, critic_fn = models

    generator_state = new_train_state(
        init_generator(init_key, generator_fn, config.data),
        generator_fn,
        config.optimizer
    )
    critic_state = new_train_state(
        init_critic(init_key, critic_fn, config.data),
        critic_fn,
        config.optimizer
    )

    # pmap over devices
    pstep_fn = jax.pmap(step_fn, axis_name="batch", in_axes=(0, 0, 0, None, 0, 0))
    peval_fn = jax.pmap(eval_fn, axis_name="batch")
    pgenerator_state = jax_utils.replicate(generator_state)
    pcritic_state = jax_utils.replicate(critic_state)

    train_metrics = []
    # start everthying finally
    cstep = 0
    logging.info(f"starting/resuming training at step: {cstep}")
    step_key, rng_key = jr.split(rng_key)
    for step, batch in zip(
        range(cstep + 1, config.training.n_steps + 1), train_iter
    ):
        train_key, val_key, sample_key = jr.split(jr.fold_in(step_key, step), 3)
        pbatch = common_utils.shard(batch)
        if step == 1 and jax.process_index() == 0:
            logging.info(f"pbatch shape: {pbatch['image'].shape}")
        do_generator_update = step % config.training.n_update_generator == 0 or step == 1
        metrics, (pcritic_state, pgenerator_state) = pstep_fn(
            jr.split(train_key, jax.device_count()),
            pcritic_state,
            pgenerator_state,
            True if do_generator_update else None,
            pbatch["image"],
            pbatch["label"],
        )
        if do_generator_update: train_metrics.append(metrics)
        is_first_or_last_step = step == config.training.n_steps or step == 1
        if (
            step % config.training.n_eval_frequency == 0
            or is_first_or_last_step
        ):
            # do evaluation loop
            val_metrics = []
            for val_idx, batch in zip(range(config.training.n_eval_batches), val_iter):
                pbatch = common_utils.shard(batch)
                metrics = peval_fn(
                    jr.split(jr.fold_in(val_key, val_idx), jax.device_count()),
                    pcritic_state,
                    pgenerator_state,
                    pbatch["image"],
                    pbatch["label"],
                )
                val_metrics.append(metrics)
            # store eval losses
            summary = metrics_to_summary(train_metrics, val_metrics)
            train_metrics = []
            if jax.process_index() == 0:
                logging.info(
                    f"loss at step {step}: "
                    f"{summary['train/critic_loss']}/{summary['train/generator_loss']}/"
                    f"{summary['val/critic_loss']}/{summary['val/generator_loss']}"
                )
            if FLAGS.usewand and jax.process_index() == 0:
                wandb.log(summary, step=step)
        if (
            step % config.training.n_sampling_frequency == 0
            and jax.process_index() == 0
        ):
            log_images(
                sample_key,
                jax_utils.unreplicate(pgenerator_state),
                step,
                model_id,
            )
    logging.info("finished training")


def plot_figures(samples):
    img_size = FLAGS.config.data.shape[0]
    n_chan = FLAGS.config.data.shape[-1]
    n_samples = samples.shape[0]
    n_row, n_col = 12, 32
    chex.assert_equal(n_samples, n_row * n_col)

    def convert_batch_to_image_grid(image_batch):
        reshaped = (
            image_batch.reshape(n_row, n_col, img_size, img_size, n_chan)
            .transpose([0, 2, 1, 3, 4])
            .reshape(n_row * img_size, n_col * img_size, n_chan)
        )
        # undo intitial scaling, i.e., map [-1, 1] -> [0, 1]
        return reshaped / 2.0 + 0.5

    samples = convert_batch_to_image_grid(samples)
    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(
        samples,
        interpolation="nearest",
        cmap="gray",
    )
    plt.axis("off")
    plt.tight_layout()
    return fig


def log_images(rng_key, generator_state, step, model_id):
    logging.info("sampling images")

    batch_size = 64
    n_samples = batch_size * 6

    @jax.jit
    def sample(rng_key, context):
        samples = generator_state.apply_fn(
            variables={"params": generator_state.params},
            rngs={"sample": rng_key},
            sample_shape=(batch_size,),
            context=context,
            is_training=False
        )
        return samples

    all_samples = []
    for i in range(n_samples // batch_size):
        if "n_classes" in FLAGS.config.data:
            sample_key, rng_key = jr.split(rng_key)
            context = jr.choice(
                sample_key,
                FLAGS.config.data.n_classes,
                (batch_size,),
                replace=True,
            )
        else:
            context = None
        sample_key, rng_key = jr.split(rng_key)
        samples = sample(sample_key, context)
        all_samples.append(samples)
    all_samples = np.concatenate(all_samples, axis=0)
    fig = plot_figures(all_samples)

    if FLAGS.usewand:
        wandb.log({"images": wandb.Image(fig)}, step=step)

    for dpi in [200]:
        fl = os.path.join(
            FLAGS.workdir, f"{model_id}-sampled-{step}-dpi-{dpi}.png"
        )
        fig.savefig(fl, dpi=dpi)


def hash_value(config):
    h = hashlib.new("sha256")
    h.update(str(config).encode("utf-8"))
    return h.hexdigest()


def init_and_log_jax_env(tm):
    logging.set_verbosity(logging.INFO)
    logging.info("file prefix: %s", tm)
    logging.info("----- Checking JAX installation ----")
    logging.info(jax.devices())
    logging.info(jax.default_backend())
    logging.info(jax.device_count())
    logging.info(xla_bridge.get_backend().platform)
    logging.info("------------------------------------")
    return tm


def main(argv):
    del argv
    config = FLAGS.config.to_dict()
    model_id = f"{hash_value(config)}"
    init_and_log_jax_env(model_id)

    if FLAGS.usewand:
        wandb.init(
            project="wgan-experiment",
            config=config,
            dir=os.path.join(FLAGS.workdir, "wandb"),
        )
        wandb.run.name = model_id

    rng_key = jr.PRNGKey(FLAGS.config.rng_key)
    data_key, train_key, rng_key = jr.split(rng_key, 3)
    train_iter, val_iter = data_loaders(
        rng_key=data_key,
        config=FLAGS.config,
        split=["train[:95%]", "train[95%:]"],
        outpath=os.path.join(FLAGS.workdir, "data"),
    )

    models, matching_fns = get_model_and_step_fn(FLAGS.config)
    train(
        train_key,
        models,
        matching_fns,
        FLAGS.config,
        train_iter,
        val_iter,
        model_id,
    )


if __name__ == "__main__":
    app.run(main)
