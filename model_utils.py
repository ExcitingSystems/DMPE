from functools import partial

import jax
import jax.numpy as jnp
import equinox as eqx
from models import NeuralEulerODE


@eqx.filter_jit
def simulate_ahead(
    model: NeuralEulerODE,
    init_obs: jnp.ndarray,
    actions: jnp.ndarray,
    tau: float
) -> jnp.ndarray:
    """Uses the given model to look ahead and simulate future observations.
    
    Args:
        model: The model to use in the simulation
        init_obs: The initial observation from which to start the simulation
        actions: The actions to apply in each step of the simulation, the length
            of the first dimension of this array determine the lenght of the
            output.
        tau: The sampling time for the model

    Returns:
        observations: The gathered observations. The shape of this is given as
            (n_actions + 1, obs_dim). That is because the first observation is
            already given through the initial observation
    """

    def body_fun(carry, action):
        obs = carry
        obs = model.step(obs, action, tau)
        return obs

    _, observations = jax.lax.scan(body_fun, init_obs, actions)
    observations = jnp.concatenate([init_obs[None, :], observations], axis=0)

    return observations


@eqx.filter_jit
def simulate_ahead_with_env(
    env,
    init_obs: jnp.ndarray,
    init_state: jnp.ndarray,
    actions: jnp.ndarray,
    env_state_normalizer: jnp.ndarray,
    action_normalizer: jnp.ndarray,
    static_params: dict
) -> jnp.ndarray:
    """Uses the given environment to look ahead and simulate future observations.
    This is used to have perfect predictions

    Args:
        model: The model to use in the simulation
        init_obs: The initial observation from which to start the simulation
        init_state: The initial state from which to start the simulation
        actions: The actions to apply in each step of the simulation, the length
            of the first dimension of this array determine the lenght of the
            output.
        env_state_normalizer: Values for state normalization
        action_normalizer: Values for action normalization
        static_params: Static parameters of the environment

    Returns:
        observations: The gathered observations. The shape of this is given as
            (n_actions + 1, obs_dim). That is because the first observation is
            already given through the initial observation
    """

    def body_fun(carry, action):
        obs, state = carry

        state = env._ode_exp_euler_step(
            state,
            action,
            env_state_normalizer,
            action_normalizer,
            static_params
        )
        obs = env.generate_observation(state)

        return (obs, state), obs

    (_, _), observations = jax.lax.scan(body_fun, (init_obs, init_state), actions)
    observations = jnp.concatenate([init_obs[None, :], observations], axis=0)

    return observations
