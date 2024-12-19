import jax
from functools import partial
from flax import nnx
from jax import random as jr


def _downsample(inputs):
    n, h, w, c = inputs.shape
    return jax.image.resize(inputs, shape=(n, h // 2, w // 2, c),
                            method="bilinear")


def _upsample(inputs):
    n, h, w, c = inputs.shape
    return jax.image.resize(inputs, shape=(n, h * 2, w * 2, c),
                            method="bilinear")


def _up_conv_block(dchan, rngs):
    conv_fn = partial(
        nnx.Conv,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_init=nnx.initializers.normal(0.02),
        padding="SAME",
        rngs=rngs
    )

    return nnx.Sequential(
        nnx.LayerNorm(dchan, rngs=rngs),
        _upsample,
        conv_fn(dchan, dchan),
        lambda x: nnx.leaky_relu(x, 0.02),
    )


def _from_latent(dlatent, dchan, base_resolution, rngs):
    return nnx.Sequential(
        nnx.Linear(
            in_features=dlatent,
            out_features=base_resolution[0] * base_resolution[1] * dchan,
            rngs=rngs,
        ),
        lambda x: nnx.leaky_relu(x, 0.02),
        lambda x: x.reshape((-1, *base_resolution, dchan)),
    )


def _to_rgb(shape, dchan, rngs):
    return nnx.Sequential(
        nnx.Conv(
            in_features=dchan,
            out_features=shape[-1],
            kernel_size=(1, 1),
            strides=(1, 1),
            kernel_init=nnx.initializers.normal(0.02),
            padding="SAME",
            rngs=rngs),
        nnx.tanh
    )


class Generator(nnx.Module):
    def __init__(self, shape, dlatent, dchan, base_resolution, rngs):
        self.rngs = rngs
        self.dlatent = dlatent
        self.model = nnx.Sequential(
            _from_latent(dlatent, dchan, base_resolution, rngs),
            _up_conv_block(dchan, rngs),
            _up_conv_block(dchan, rngs),
            _up_conv_block(dchan, rngs),
            _to_rgb(shape, dchan, rngs)
        )

    def __call__(self, sample_shape=(), context=None, **kwargs):
        if "latents" in kwargs:
            latents = kwargs["latents"]
        else:
            latents = self.sample_latent(sample_shape, context)
        outs = self.model(latents)
        return outs

    def sample_latent(self, sample_shape=(), context=None, **kwargs):
        rng_key = self.rngs.noise()
        return jr.normal(rng_key, sample_shape + (self.dlatent,))


def _down_conv_block(dchan, rngs):
    conv_fn = partial(
        nnx.Conv,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_init=nnx.initializers.normal(0.02),
        padding="SAME",
        rngs=rngs
    )

    return nnx.Sequential(
        nnx.LayerNorm(dchan, rngs=rngs),
        _downsample,
        conv_fn(dchan, dchan),
        lambda x: nnx.leaky_relu(x, 0.02),
    )


def _from_rgb(shape, dchan, rngs):
    return nnx.Sequential(
            nnx.Conv(
                in_features=shape[-1],
                out_features=dchan,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="SAME",
                rngs=rngs
            ),
            nnx.tanh,
    )


def _to_flat(dchan, base_resolution, rngs):
    n_in_features = dchan * base_resolution[0] * base_resolution[1]
    return nnx.Sequential(
        lambda x: x.reshape(x.shape[0], -1),
        nnx.Linear(n_in_features, 1, rngs=rngs)
    )


class Critic(nnx.Module):
    def __init__(self, shape, dlatent, dchan, base_resolution, rngs):
        self.rngs = rngs
        self.dlatent = dlatent
        self.model = nnx.Sequential(
            _from_rgb(shape, dchan , rngs),
            _down_conv_block(dchan, rngs),
            _down_conv_block(dchan, rngs),
            _down_conv_block(dchan , rngs),
            _to_flat(dchan, base_resolution, rngs),
        )

    def __call__(self, inputs, **kwargs):
        outs = self.model(inputs)
        return outs


def make_model(shape, config, rng_key):
    return (
        Generator(shape=shape, rngs=nnx.Rngs(rng_key, noise=rng_key+1), **config),
        Critic(shape=shape, rngs=nnx.Rngs(rng_key + 2), **config),
    )
