import jax
import jax.numpy as jnp
import equinox as eqx
from exciting_exciting_systems.models import NeuralEulerODE


@eqx.filter_jit
def simulate_ahead(model: NeuralEulerODE, init_obs: jnp.ndarray, actions: jnp.ndarray, tau: float) -> jnp.ndarray:
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
    return model(init_obs, actions, tau)


@eqx.filter_jit
def simulate_ahead_with_env(
    env,
    init_obs: jnp.ndarray,
    init_state: jnp.ndarray,
    actions: jnp.ndarray,
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

        state = env._ode_solver_step(
            state, action * env.env_properties.action_constraints.torque, env.env_properties.static_params
        )
        obs = env.generate_observation(state, env.env_properties.physical_constraints)
        return (obs, state), obs

    (_, _), observations = jax.lax.scan(body_fun, (init_obs, init_state), actions)
    observations = jnp.concatenate([init_obs[None, :], observations], axis=0)

    return observations
