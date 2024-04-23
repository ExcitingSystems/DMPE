from functools import partial

import jax
import jax.numpy as jnp

import exciting_environments as excenvs


@partial(jax.jit, static_argnums=(0, 1))
def simulate_ahead(
    model: excenvs.core_env.CoreEnvironment,  # typehint for the time being...
    n_steps: int,
    init_obs: jnp.ndarray,
    init_state: jnp.ndarray,
    actions: jnp.ndarray,
    env_state_normalizer,
    action_normalizer,
    static_params
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

    def body_fun(carry, action):
        obs, state = carry

        state = model._ode_exp_euler_step(
            state,
            action,
            env_state_normalizer,
            action_normalizer,
            static_params
        )
        next_obs = model.generate_observation(state)

        return (obs, state), next_obs

    (obs, state), observations = jax.lax.scan(body_fun, (init_obs, init_state), actions[:-1, :])
    observations = jnp.concatenate([obs[None, :], observations], axis=0)

    return observations
