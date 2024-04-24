from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(0, 1))
def generate_constant_action(n_steps, batch_size, key):
    actions = jax.random.uniform(key, shape=(batch_size, 1, 1), minval=-1, maxval=1)
    actions = jnp.repeat(actions, repeats=n_steps, axis=1)
    return actions


def aprbs_single_batch(len, t_min, t_max, key):
    t = 0
    sig = []
    while t < len:
        steps_key, value_key, key = jax.random.split(key, 3)

        t_step = jax.random.randint(steps_key, shape=(1,), minval=t_min, maxval=t_max)

        sig.append(jnp.ones(t_step) * jax.random.uniform(value_key, shape=(1,), minval=-1, maxval=1))
        t += t_step.item()

    return jnp.hstack(sig)[:len]


def aprbs(n_steps, batch_size, t_min, t_max, key):
    actions = []
    for _ in range(batch_size):
        subkey, key = jax.random.split(key)
        actions.append(aprbs_single_batch(n_steps, t_min, t_max, subkey)[..., None])
    return jnp.stack(actions, axis=0)
