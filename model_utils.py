from functools import partial

import jax
import jax.numpy as jnp

import exciting_environments as excenvs


@partial(jax.jit, static_argnums=(0, 1))
def simulate_ahead(
    model: excenvs.core_env.CoreEnvironment,  # typehint for the time being...
    n_steps: int,
    obs: jnp.ndarray,
    state: jnp.ndarray,
    actions: jnp.ndarray
) -> jnp.ndarray:
    """Uses the given model to look ahead and simulate future observations.
    
    Args:
        model: The model to use in the simulation
        n_steps: The number of steps to simulate into the future
        obs: The current observations from which to start the simulation
        state: The current state from which to start the simulation
        actions: The actions to apply in each step of the simulation

    Returns:
        observations: The gathered observations
    """

    batch_size, obs_dim = obs.shape
    observations = jnp.zeros([batch_size, n_steps, obs_dim])
    observations = observations.at[:, 0, :].set(obs)

    if isinstance(model, excenvs.core_env.CoreEnvironment):
        step = lambda action, state: model.step(action, state)
    else:
        step = lambda action, state: model(action, state)

    def body_fun(n, carry):
        obs, state, observations = carry

        action = actions[:, n, :]
        obs, _, _, _, state = step(action, state)
        observations = observations.at[:, n, :].set(obs)

        return (obs, state, observations)

    obs, state, observations = jax.lax.fori_loop(lower=1, upper=n_steps, body_fun=body_fun, init_val=(obs, state, observations))

    return observations
