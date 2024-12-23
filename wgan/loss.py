import dataclasses
from functools import partial

import numpy as np
from flax import nnx
from jax import numpy as jnp
from jax import random as jr


@dataclasses.dataclass
class WGANGPConfig:
    n_update_generator: int = 5
    lamb: float = 10


class WGANGP:
    def __new__(cls, config: WGANGPConfig):
        def generator_loss_fn(
            generator_fn, critic_fn, rng_key, inputs, context=None, **kwargs
        ):
            synthetic_data = generator_fn((inputs.shape[0],), context=context)
            preds = critic_fn(synthetic_data)
            return -jnp.mean(preds)

        @partial(nnx.vmap, in_axes=(None, 0, 0))
        @partial(nnx.grad, argnums=1)
        def _critic_forward(model_fn, inputs, context):
            value = model_fn(inputs[None], context=None)
            return value[0, 0]

        def critic_loss_fn(
            critic_fn, generator_fn, rng_key, inputs, context=None, **kwargs
        ):
            sample_key, rng_key = jr.split(rng_key)
            synthetic_data = generator_fn((inputs.shape[0],), context=context)
            preds_synthetic = critic_fn(synthetic_data, context=context)
            preds_inputs = critic_fn(inputs, context=context)

            sample_key, rng_key = jr.split(rng_key)
            new_shape = tuple(np.ones(inputs.ndim - 1, dtype=np.int32).tolist())
            epsilon = jr.uniform(
                sample_key, shape=(inputs.shape[0], *new_shape)
            )
            data_mix = inputs * epsilon + synthetic_data * (1 - epsilon)

            gradients = _critic_forward(critic_fn, data_mix, context)
            gradients = gradients.reshape((gradients.shape[0], -1))
            grad_norm = jnp.linalg.norm(gradients, axis=1)
            grad_penalty = ((grad_norm - 1) ** 2).mean()

            loss = (
                -jnp.mean(preds_inputs)
                + jnp.mean(preds_synthetic)
                + config.lamb * grad_penalty
            )
            return loss

        @nnx.jit
        def step_fn(
            rng_key,
            model_fns,
            optimizer_fns,
            step,
            inputs,
            context,
            metrics,
            **kwargs,
        ):
            generator_fn, critic_fn = model_fns
            g_optimizer, c_optimizer = optimizer_fns

            grad_key, rng_key = jr.split(rng_key)
            critic_fn.train()
            generator_fn.eval()
            grad_fn = nnx.value_and_grad(critic_loss_fn)
            loss_c, grads = grad_fn(
                critic_fn, generator_fn, grad_key, inputs, context
            )
            c_optimizer.update(grads)

            loss_g = None
            if step is not None:
                grad_key, rng_key = jr.split(rng_key)
                generator_fn.train()
                critic_fn.eval()
                grad_fn = nnx.value_and_grad(generator_loss_fn)
                loss_g, grads = grad_fn(
                    generator_fn, critic_fn, grad_key, inputs, context
                )
                g_optimizer.update(grads)
                metrics.update(critic_loss=loss_c, generator_loss=loss_g)

            return {"loss_c": loss_c, "loss_g": loss_g}

        @nnx.jit
        def eval_fn(rng_key, model_fns, inputs, context, metrics, **kwargs):
            generator_fn, critic_fn = model_fns
            loss_c_key, loss_g_key = jr.split(rng_key)
            critic_fn.eval()
            generator_fn.eval()
            loss_c = critic_loss_fn(
                critic_fn, generator_fn, loss_c_key, inputs, context
            )
            loss_g = generator_loss_fn(
                generator_fn, critic_fn, loss_g_key, inputs, context
            )
            metrics.update(critic_loss=loss_c, generator_loss=loss_g)

            return {"loss_c": loss_c, "loss_g": loss_g}

        return step_fn, eval_fn
