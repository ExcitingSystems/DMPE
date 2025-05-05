from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(0, 1))
def generate_constant_action(n_steps: int, batch_size: int, key: jax.random.PRNGKey) -> jax.Array:
    """Randomly draws an action and repeats it for 'n_steps'."""
    actions = jax.random.uniform(key, shape=(batch_size, 1, 1), minval=-1, maxval=1)
    actions = jnp.repeat(actions, repeats=n_steps, axis=1)
    return actions


def aprbs_single_batch(len: int, t_min: float, t_max: float, key: jax.random.PRNGKey) -> jax.Array:
    """Creates an amplitude modulated pseudorandom binary sequence (APRBS) in 1d and for 1 batch,
    i.e. without a batch dimension.

    Args:
        len (int): Length of the signal.
        t_min (float): Minimum hold time of an amplitude.
        t_max (float): Maximum hold time of an amplitude
        key (jax.random.PRNGKey): Random key for JAX random sampling.
    """

    t = 0
    sig = []
    while t < len:
        steps_key, value_key, key = jax.random.split(key, 3)

        t_step = jax.random.randint(steps_key, shape=(1,), minval=t_min, maxval=t_max)

        sig.append(jnp.ones(t_step) * jax.random.uniform(value_key, shape=(1,), minval=-1, maxval=1))
        t += t_step.item()

    return jnp.hstack(sig)[:len]


def aprbs(n_steps: int, batch_size: int, t_min: float, t_max: float, key: jax.random.PRNGKey) -> jax.Array:
    """Creates an amplitude modulated pseudorandom binary sequence (APRBS) in 1d where 'batch_size'
    sequences are created independently next to each other.

    Args:
        n_steps (int): Length of the signal.
        batch_size (int): Number of batches.
        t_min (float): Minimum hold time of an amplitude.
        t_max (float): Maximum hold time of an amplitude
        key (jax.random.PRNGKey): Random key for JAX random sampling.
    """
    actions = []
    for _ in range(batch_size):
        subkey, key = jax.random.split(key)
        actions.append(aprbs_single_batch(n_steps, t_min, t_max, subkey)[..., None])
    return jnp.stack(actions, axis=0)
