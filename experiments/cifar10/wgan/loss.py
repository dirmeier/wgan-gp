import dataclasses
from functools import partial

import jax
import numpy as np
from flax import linen as nn
from jax import numpy as jnp
from jax import random as jr


@dataclasses.dataclass
class WGANGPConfig:
    n_update_generator: int = 5
    lamb: float = 10


class WGANGP:
    def __new__(cls, generator_fn, critic_fn, config: WGANGPConfig):
        def generator_loss_fn(
            params, generator_state, critic_state, rng_key, inputs, context, is_training
        ):
            synthetic_data = generator_fn(
                variables={"params": params},
                rngs={"sample": rng_key},
                sample_shape=(inputs.shape[0],),
                context=context,
                is_training=is_training
            )
            preds = critic_fn(
                variables={"params": critic_state.params},
                rngs=None,
                inputs=synthetic_data,
                context=context,
                is_training=is_training
            )
            return -jnp.mean(preds)

        @partial(jax.vmap, in_axes=(None, None, 0, 0))
        @partial(jax.grad, argnums=2)
        def _critic_forward(params, critic_state, inputs, context):
            value = critic_fn(
                variables={"params": params},
                rngs=None,
                inputs=inputs[None],
                context=None,
                is_training=True
            )
            return value[0, 0]

        def critic_loss_fn(
            params, critic_state, generator_state, rng_key, inputs, context, is_training
        ):
            sample_key, rng_key = jr.split(rng_key)
            synthetic_data = generator_fn(
                variables={"params": generator_state.params},
                rngs={"sample": rng_key},
                sample_shape=(inputs.shape[0],),
                context=context,
                is_training=is_training
            )
            preds_synthetic = critic_fn(
                variables={"params": params},
                rngs=None,
                inputs=synthetic_data,
                context=context,
                is_training=is_training
            )
            preds_inputs = critic_fn(
                variables={"params": params},
                rngs=None,
                inputs=inputs,
                context=context,
                is_training=is_training
            )

            sample_key, rng_key = jr.split(rng_key)
            new_shape = tuple(np.ones(inputs.ndim - 1, dtype=np.int32).tolist())
            epsilon = jr.uniform(
                sample_key, shape=(inputs.shape[0], *new_shape)
            )
            data_mix = inputs * epsilon + synthetic_data * (1 - epsilon)

            gradients = _critic_forward(params, critic_state, data_mix, context)
            gradients = gradients.reshape((gradients.shape[0], -1))
            grad_norm = jnp.linalg.norm(gradients, axis=1)
            grad_penalty = ((grad_norm - 1) ** 2).mean()

            loss = (
                -jnp.mean(preds_inputs)
                + jnp.mean(preds_synthetic)
                + config.lamb * grad_penalty
            )
            return loss

        def step_fn(
            rng_key,
            critic_state,
            generator_state,
            step,
            inputs,
            context,
        ):
            grad_key, rng_key = jr.split(rng_key)
            grad_fn = jax.value_and_grad(critic_loss_fn)
            loss_c, grads = grad_fn(
                critic_state.params, critic_state, generator_state, grad_key, inputs, context, True
            )
            loss_c = jax.lax.pmean(loss_c, axis_name="batch")
            grads = jax.lax.pmean(grads, axis_name="batch")
            new_critic_state = critic_state.apply_gradients(grads=grads)

            loss_g = None
            new_generator_state = generator_state
            if step is not None:
                grad_key, rng_key = jr.split(rng_key)
                grad_fn = jax.value_and_grad(generator_loss_fn)
                loss_g, grads = grad_fn(
                    generator_state.params, generator_state, critic_state, grad_key, inputs, context, True
                )
                loss_g = jax.lax.pmean(loss_g, axis_name="batch")
                grads = jax.lax.pmean(grads, axis_name="batch")
                new_generator_state = generator_state.apply_gradients(grads=grads)

            return {"critic_loss": loss_c, "generator_loss": loss_g}, (new_critic_state, new_generator_state)

        @jax.jit
        def eval_fn(rng_key,     critic_state, generator_state, inputs, context):
            loss_c_key, loss_g_key = jr.split(rng_key)
            loss_c = critic_loss_fn(
                critic_state.params, critic_state, generator_state, loss_c_key, inputs, context, False
            )
            loss_g = generator_loss_fn(
                generator_state.params, generator_state, critic_state, loss_g_key, inputs, context, False
            )

            return {"critic_loss": loss_c, "generator_loss": loss_g}

        return step_fn, eval_fn
