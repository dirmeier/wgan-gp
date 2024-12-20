from functools import partial

from flax import linen as nn
from jax import random as jr
import jax

# class Generator(nn.Module):
#     shape: tuple
#     dlatent: int
#     dhid: int

#     def setup(self):
#         conv_transpose = partial(
#             nn.ConvTranspose,
#             kernel_init=nn.initializers.normal(0.02),
#         )
#         self.model = nn.Sequential([
#             conv_transpose(8 * self.dhid, (4, 4), (1, 1), padding="VALID"),
#             nn.LayerNorm(),
#             nn.relu,

#             conv_transpose(4 * self.dhid, (4, 4), (2, 2), padding="SAME"),
#             nn.LayerNorm(),
#             nn.relu,

#             conv_transpose(2 * self.dhid, (4, 4), (2, 2), padding="SAME"),
#             nn.LayerNorm(),
#             nn.relu,

#             conv_transpose(self.dhid, (4, 4), (2, 2), padding="SAME"),
#             nn.LayerNorm(),
#             nn.relu,

#             conv_transpose(self.shape[-1], (4, 4), (1, 1), padding="SAME"),
#             nn.tanh,
#         ])

#     def __call__(self, sample_shape=(), context=None, **kwargs):
#         if "latents" in kwargs:
#             latents = kwargs["latents"]
#         else:
#             latents = self.sample_latent(sample_shape, context)
#         hidden = latents.reshape(-1, 1, 1, self.dlatent)
#         outs = self.model(hidden)
#         return outs

#     def sample_latent(self, sample_shape=(), context=None, **kwargs):
#         rng_key = self.make_rng("sample")
#         return jr.normal(rng_key, sample_shape + (self.dlatent,))


# class Critic(nn.Module):
#     shape: tuple
#     dhid: int

#     def setup(self):
#         conv = partial(
#             nn.Conv, kernel_init=nn.initializers.normal(0.02)
#         )
#         self.model = nn.Sequential([
#             conv(self.dhid, (4, 4), (2, 2), padding="SAME"),
#             lambda x: nn.leaky_relu(x, 0.2),

#             conv(2 * self.dhid, (4, 4), (2, 2), padding="SAME"),
#             nn.LayerNorm(),
#             lambda x: nn.leaky_relu(x, 0.2),

#             conv(4 * self.dhid, (4, 4), (2, 2), padding="SAME"),
#             nn.LayerNorm(),
#             lambda x: nn.leaky_relu(x, 0.2),

#             conv( 1, (4, 4), (1, 1), padding="VALID"),
#         ])

#     def __call__(self, inputs, **kwargs):
#         outs = self.model(inputs)
#         return outs.reshape(inputs.shape[0], -1)


def _downsample(inputs):
    n, h, w, c = inputs.shape
    return jax.image.resize(inputs, shape=(n, h // 2, w // 2, c), method="bilinear")


def _upsample(inputs):
    n, h, w, c = inputs.shape
    return jax.image.resize(inputs, shape=(n, h * 2, w * 2, c), method="bilinear")


class Generator(nn.Module):
    shape: tuple
    dlatent: int
    dchan: int
    base_resolution: tuple

    def setup(self):
        conv = partial(
            nn.Conv,
            kernel_size=(3, 3),
            kernel_init=nn.initializers.normal(0.02),
            strides=(1, 1),
            padding="SAME",
        )
        self.model = nn.Sequential([
            nn.Dense(self.base_resolution[0] * self.base_resolution[1] * self.dchan),
            nn.leaky_relu,
            lambda x: x.reshape((-1, *self.base_resolution, self.dchan)),
            #
            nn.GroupNorm(32),
            _upsample,
            conv(self.dchan),
            nn.leaky_relu,
            #
            nn.GroupNorm(32),
            _upsample,
            conv(self.dchan),
            nn.leaky_relu,
            #
            nn.GroupNorm(32),
            _upsample,
            conv(self.dchan),
            nn.leaky_relu,
            #
            nn.Conv(self.shape[-1], (1, 1), (1, 1), padding="SAME"),
            nn.tanh
        ])

    def __call__(self, sample_shape=(), context=None, **kwargs):
        if "latents" in kwargs:
            latents = kwargs["latents"]
        else:
            latents = self.sample_latent(sample_shape, context)
        outs = self.model(latents)
        return outs

    def sample_latent(self, sample_shape=(), context=None, **kwargs):
        rng_key = self.make_rng("sample")
        return jr.normal(rng_key, sample_shape + (self.dlatent,))


class Critic(nn.Module):
    shape: tuple
    dlatent: int
    dchan: int
    base_resolution: tuple

    def setup(self):
        conv = partial(
            nn.Conv,
            kernel_size=(3, 3),
            kernel_init=nn.initializers.normal(0.02),
            strides=(1, 1),
            padding="SAME",
        )
        self.model = nn.Sequential([
            nn.Conv(self.dchan, (1, 1), (1, 1), padding="SAME"),
            #
            _downsample,
            nn.GroupNorm(32),
            conv(self.dchan),
            nn.leaky_relu,
            #
            _downsample,
            nn.GroupNorm(32),
            conv(self.dchan),
            nn.leaky_relu,
            #
            _downsample,
            nn.GroupNorm(32),
            conv(self.dchan),
            nn.leaky_relu,
            #
            lambda x: x.reshape(x.shape[0], -1),
            nn.Dense(1)
        ])

    def __call__(self, inputs, **kwargs):
        outs = self.model(inputs)
        return outs.reshape(inputs.shape[0], -1)


def make_model(shape, config):
    return (
        Generator(shape=shape, **config),
        Critic(shape=shape ,  **config),
    )
