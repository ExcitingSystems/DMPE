import json
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten

import equinox as eqx

import exciting_environments as excenvs
from dmpe.models import NeuralEulerODE


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

    action_denormalizer = jnp.array(tree_flatten(env.env_properties.action_constraints)[0], dtype=jnp.float32)

    def body_fun(carry, action):
        obs, state = carry

        state = env._ode_solver_step(state, action * action_denormalizer, env.env_properties.static_params)
        obs = env.generate_observation(state, env.env_properties.physical_constraints)
        return (obs, state), obs

    (_, last_state), observations = jax.lax.scan(body_fun, (init_obs, init_state), actions)
    observations = jnp.concatenate([init_obs[None, :], observations], axis=0)

    return observations, last_state


class ModelEnvWrapperFluidTank(eqx.Module):
    env: excenvs.CoreEnvironment

    def __call__(self, init_obs, actions, tau):

        # assumes full observability
        # assumes tau == env.tau
        # practical denormilization for fluid tank yikes
        # TODO: sort this out, error-prone
        init_state = self.env.State(
            physical_state=self.env.PhysicalState((init_obs + 1) * 3 / 2),
            PRNGKey=None,
            optional=None,
        )

        observations, _ = simulate_ahead_with_env(self.env, init_obs, init_state, actions)
        return observations


class ModelEnvWrapperPendulum(eqx.Module):
    env: excenvs.CoreEnvironment

    def __call__(self, init_obs, actions, tau):

        # assumes full observability
        # assumes tau == env.tau
        # practical denormilization for fluid tank yikes
        # TODO: sort this out, error-prone
        init_state = self.env.State(
            physical_state=self.env.PhysicalState(
                theta=init_obs[..., 0] * self.env.env_properties.physical_constraints.theta,
                omega=init_obs[..., 1] * self.env.env_properties.physical_constraints.omega,
            ),
            PRNGKey=None,
            optional=None,
        )

        observations, _ = simulate_ahead_with_env(self.env, init_obs, init_state, actions)
        return observations


class ModelEnvWrapperCartPole(eqx.Module):
    env: excenvs.CoreEnvironment

    def __call__(self, init_obs, actions, tau):

        init_state = self.env.State(
            physical_state=self.env.PhysicalState(
                init_obs[..., 0] * self.env.env_properties.physical_constraints.deflection,
                init_obs[..., 1] * self.env.env_properties.physical_constraints.velocity,
                init_obs[..., 2] * self.env.env_properties.physical_constraints.theta,
                init_obs[..., 3] * self.env.env_properties.physical_constraints.omega,
            ),
            PRNGKey=None,
            optional=None,
        )
        observations, _ = simulate_ahead_with_env(self.env, init_obs, init_state, actions)
        return observations


class ModelWrapperMassSpringDamper(eqx.Module):
    env: excenvs.CoreEnvironment

    def __call__(self, init_obs, actions, tau):
        init_state = self.env.State(
            physical_state=self.env.PhysicalState(
                init_obs[..., 0] * self.env.env_properties.physical_constraints.deflection,
                init_obs[..., 1] * self.env.env_properties.physical_constraints.velocity,
            ),
            PRNGKey=None,
            optional=None,
        )
        observations, _ = simulate_ahead_with_env(self.env, init_obs, init_state, actions)
        return observations


def save_model(filename, hyperparams, model):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load_model(filename, model_class):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        hyperparams["key"] = jnp.array(hyperparams["key"], dtype=jnp.uint32)
        model = model_class(**hyperparams)
        return eqx.tree_deserialise_leaves(f, model)
