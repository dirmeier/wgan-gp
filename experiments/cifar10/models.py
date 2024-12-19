from functools import partial

from flax import nnx
from jax import random as jr


class Generator(nnx.Module):
    def __init__(self, shape, dlatent, dhid, rngs, **kwargs):
        self.dlatent = dlatent
        self.dhid = dhid
        self.shape = shape
        self.rngs = rngs
        conv_transpose = partial(
            nnx.ConvTranspose,
            kernel_init=nnx.initializers.normal(0.02),
            rngs=rngs,
        )

        self.model = nnx.Sequential(
            conv_transpose(dlatent, 8 * dhid, (4, 4), (1, 1), padding="VALID"),
            nnx.LayerNorm(8 * dhid, rngs=rngs),
            nnx.relu,
            conv_transpose(8 * dhid, 4 * dhid, (4, 4), (2, 2), padding="SAME"),
            nnx.LayerNorm(4 * dhid, rngs=rngs),
            nnx.relu,
            conv_transpose(4 * dhid, 2 * dhid, (4, 4), (2, 2), padding="SAME"),
            nnx.LayerNorm(2 * dhid, rngs=rngs),
            nnx.relu,
            conv_transpose(2 * dhid, dhid, (4, 4), (2, 2), padding="SAME"),
            nnx.LayerNorm(dhid, rngs=rngs),
            nnx.relu,
            conv_transpose(dhid, shape[-1], (4, 4), (1, 1), padding="SAME"),
            nnx.tanh,
        )

    def __call__(self, sample_shape=(), context=None, **kwargs):
        if "latents" in kwargs:
            latents = kwargs["latents"]
        else:
            latents = self.sample_latent(sample_shape, context)
        hidden = latents.reshape(-1, 1, 1, self.dlatent)
        outs = self.model(hidden)
        return outs

    def sample_latent(self, sample_shape=(), context=None, **kwargs):
        rng_key = self.rngs["sample"]()
        return jr.normal(rng_key, sample_shape + (self.dlatent,))


class Critic(nnx.Module):
    def __init__(self, shape, dhid, rngs, **kwargs):
        self.shape = shape

        conv = partial(
            nnx.Conv, kernel_init=nnx.initializers.normal(0.02), rngs=rngs
        )
        self.model = nnx.Sequential(
            conv(shape[-1], dhid, (4, 4), (2, 2), padding="SAME"),
            lambda x: nnx.leaky_relu(x, 0.2),
            conv(dhid, 2 * dhid, (4, 4), (2, 2), padding="SAME"),
            nnx.LayerNorm(2 * dhid, rngs=rngs),
            lambda x: nnx.leaky_relu(x, 0.2),
            conv(2 * dhid, 4 * dhid, (4, 4), (2, 2), padding="SAME"),
            nnx.LayerNorm(
                4 * dhid,
                rngs=rngs,
            ),
            lambda x: nnx.leaky_relu(x, 0.2),
            conv(4 * dhid, 1, (4, 4), (1, 1), padding="VALID"),
        )

    def __call__(self, inputs, **kwargs):
        outs = self.model(inputs)
        return outs.reshape(inputs.shape[0], -1)


def make_model(shape, config, rngs):
    return (
        Generator(rngs=rngs, shape=shape, **config),
        Critic(rngs=rngs, shape=shape, **config),
    )
