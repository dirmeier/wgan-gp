import hashlib
import os

import chex
import jax
import matplotlib.pyplot as plt
import numpy as np
import wandb
from absl import app, flags, logging
from dataloader import data_loaders
from flax import nnx
from jax import random as jr
from jax.experimental import mesh_utils
from jax.lib import xla_bridge
from ml_collections import config_flags

from checkpointer import get_optimizer
from models import make_model
from wgan.loss import WGANGP, WGANGPConfig

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "model configuration")
flags.DEFINE_string("workdir", None, "work directory")
flags.DEFINE_bool("usewand", False, "use wandb for logging")
flags.mark_flags_as_required(["workdir", "config"])


def get_models(config):
    return make_model(
        shape=config.data.shape, config=config.nn, rng_key=config.rng_key
    )


def get_model_and_step_fn(config):
    models = get_models(config)
    step_fn, eval_fn = WGANGP(WGANGPConfig(config.training.n_update_generator))
    return models, (step_fn, eval_fn)


def train(
    rng_key, models, matching_fns, config, train_iter, val_iter, model_id
):
    init_key, rng_key = jr.split(rng_key)
    # step and eval fns, and generator and discriminator
    pstep_fn, peval_fn = matching_fns
    generator_fn, critic_fn = models

    # sharding
    num_devices = jax.local_device_count()
    mesh = jax.sharding.Mesh(
        mesh_utils.create_device_mesh((num_devices,)), ("data",)
    )
    model_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))

    # optimizers
    g_optimizer = get_optimizer(config.optimizer, generator_fn)
    c_optimizer = get_optimizer(config.optimizer, critic_fn)

    # replicate state
    state = nnx.state((generator_fn, critic_fn, g_optimizer, c_optimizer))
    state = jax.device_put(state, model_sharding)
    nnx.update((generator_fn, critic_fn, g_optimizer, c_optimizer), state)

    # metrics
    metrics_history = {}
    metrics = nnx.MultiMetric(
        critic_loss=nnx.metrics.Average("critic_loss"),
        generator_loss=nnx.metrics.Average("generator_loss"),
    )

    # start everything finally
    step_key, rng_key = jr.split(rng_key)
    for step, batch in zip(range(1, config.training.n_steps + 1), train_iter):
        train_key, val_key, sample_key = jr.split(jr.fold_in(step_key, step), 3)
        pimages, plabels = jax.device_put((batch["image"], batch["label"]), data_sharding)
        if step == 1:
            logging.info(f"shape flat: {batch['image'].shape}")
            logging.info(f"shape sharded: {pimages.shape}")
        do_generator_update = step % config.training.n_update_generator == 0 or step == 1
        pstep_fn(
            train_key,
            (generator_fn, critic_fn),
            (g_optimizer, c_optimizer),
            True if do_generator_update else None,
            inputs=pimages,
            context=plabels,
            metrics=metrics,
        )
        is_first_or_last_step = step == config.training.n_steps or step == 1
        if (
            step % config.training.n_eval_frequency == 0
            or is_first_or_last_step
        ):
            # store training losses
            for metric, value in metrics.compute().items():
                metrics_history[f"train_{metric}"] = value
            metrics.reset()

            # do evaluation loop
            for val_idx, batch in zip(
                range(config.training.n_eval_batches), val_iter
            ):
                pimages, plabels = jax.device_put((batch["image"], batch["label"]), data_sharding)
                peval_fn(
                    jr.fold_in(val_key, val_idx),
                    (generator_fn, critic_fn),
                    inputs=pimages,
                    context=plabels,
                    metrics=metrics,
                )

            # store eval losses
            for metric, value in metrics.compute().items():
                metrics_history[f"val_{metric}"] = value
            metrics.reset()

            if jax.process_index() == 0:
                logging.info(
                    f"loss at step {step}: "
                    f"{metrics_history['train_critic_loss']}/{metrics_history['train_generator_loss']}/"
                    f"{metrics_history['val_critic_loss']}/{metrics_history['val_generator_loss']}"
                )
            if FLAGS.usewand and jax.process_index() == 0:
                wandb.log(metrics_history, step=step)
        if (
            step % config.training.n_sampling_frequency == 0
            and jax.process_index() == 0
        ):
            # unreplicate state
            gen, _ = get_models(config)
            nnx.update(gen, jax.device_get(nnx.state(generator_fn)))
            log_images(sample_key, gen, step, model_id)
    logging.info("finished training")


def plot_figures(samples):
    img_size = FLAGS.config.data.shape[0]
    n_chan = FLAGS.config.data.shape[2]
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


def log_images(rng_key, generator_fn, step, model_id):
    logging.info("sampling images")

    batch_size = 64
    n_samples = batch_size * 6

    @nnx.jit
    def sample(rng_key, context):
        latents = jr.normal(rng_key, (batch_size, FLAGS.config.nn.dlatent))
        samples = generator_fn(
            latents=latents,
            context=context,
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
        pt = os.path.join(FLAGS.workdir, "figures")
        if not os.path.exists(pt): os.mkdir(pt)
        fl = os.path.join(pt, f"{model_id}-sampled-{step}-dpi-{dpi}.png")
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
