from functools import partial

from flax import linen as nn
from jax import random as jr


class Generator(nn.Module):
    shape: tuple
    dlatent: int
    dhid: int

    def setup(self):
        conv_transpose = partial(
            nn.ConvTranspose,
            kernel_init=nn.initializers.normal(0.02),
        )
        self.model = nn.Sequential([
            conv_transpose(8 * self.dhid, (4, 4), (1, 1), padding="VALID"),
            nn.LayerNorm(),
            nn.relu,

            conv_transpose(4 * self.dhid, (4, 4), (2, 2), padding="SAME"),
            nn.LayerNorm(),
            nn.relu,

            conv_transpose(2 * self.dhid, (4, 4), (2, 2), padding="SAME"),
            nn.LayerNorm(),
            nn.relu,

            conv_transpose(self.dhid, (4, 4), (2, 2), padding="SAME"),
            nn.LayerNorm(),
            nn.relu,

            conv_transpose(self.shape[-1], (4, 4), (1, 1), padding="SAME"),
            nn.tanh,
        ])

    def __call__(self, sample_shape=(), context=None, **kwargs):
        if "latents" in kwargs:
            latents = kwargs["latents"]
        else:
            latents = self.sample_latent(sample_shape, context)
        hidden = latents.reshape(-1, 1, 1, self.dlatent)
        outs = self.model(hidden)
        return outs

    def sample_latent(self, sample_shape=(), context=None, **kwargs):
        rng_key = self.make_rng("sample")
        return jr.normal(rng_key, sample_shape + (self.dlatent,))


class Critic(nn.Module):
    shape: tuple
    dhid: int

    def setup(self):
        conv = partial(
            nn.Conv, kernel_init=nn.initializers.normal(0.02)
        )
        self.model = nn.Sequential([
            conv(self.dhid, (4, 4), (2, 2), padding="SAME"),
            lambda x: nn.leaky_relu(x, 0.2),

            conv(2 * self.dhid, (4, 4), (2, 2), padding="SAME"),
            nn.LayerNorm(),
            lambda x: nn.leaky_relu(x, 0.2),

            conv(4 * self.dhid, (4, 4), (2, 2), padding="SAME"),
            nn.LayerNorm(),
            lambda x: nn.leaky_relu(x, 0.2),

            conv( 1, (4, 4), (1, 1), padding="VALID"),
        ])

    def __call__(self, inputs, **kwargs):
        outs = self.model(inputs)
        return outs.reshape(inputs.shape[0], -1)


def make_model(shape, config):
    return (
        Generator(shape=shape, dlatent=config.dlatent, dhid=config.dhid),
        Critic(shape=shape , dhid=config.dhid),
    )
