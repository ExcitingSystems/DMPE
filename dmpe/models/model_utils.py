import json
import pathlib

import jax
import jax.numpy as jnp
import equinox as eqx

import exciting_environments as excenvs
from dmpe.models.models import NeuralEulerODE


@eqx.filter_jit
def simulate_ahead(
    model: NeuralEulerODE,
    init_obs: jax.Array,
    actions: jax.Array,
    tau: float,
) -> jax.Array:
    """Uses the given model to look ahead and simulate future observations.

    Args:
        model (eqx.Module): The model to use in the simulation
        init_obs (jax.Array): The initial observation from which to start the simulation
        actions (jax.Array): The actions to apply in each step of the simulation, the length
            of the first dimension of this array determines the length of the output.
        tau (float): The sampling time for the model

    Returns:
        observations (jax.Array): The predicted observations. The shape of this is given as
            (n_actions + 1, obs_dim). That is because the first observation is
            already given through the initial observation
    """
    return model(init_obs, actions, tau)


@eqx.filter_jit
def simulate_ahead_with_env(
    env: excenvs.CoreEnvironment,
    init_obs: jax.Array,
    init_state: excenvs.CoreEnvironment.State,
    actions: jax.Array,
) -> jax.Array:
    """Uses the given environment to look ahead and simulate future observations.
    This is used to have perfect predictions.

    Args:
        env (excenvs.CoreEnvironment): The env used for the simulation
        init_obs (jax.Array): The initial observation from which to start the simulation
        init_state (excenvs.CoreEnvironment.State): The initial state from which to start
            the simulation
        actions (jax.Array): The actions to apply in each step of the simulation, the length
            of the first dimension of this array determine the length of the output.

    Returns:
        observations (jax.Array): The simulated observations. The shape of this is given as
            (n_actions + 1, obs_dim). That is because the first observation is already given
            through the initial observation
    """

    def body_fun(carry, action):
        obs, state = carry

        obs, state = env.step(state, action, env.env_properties)
        return (obs, state), obs

    (_, last_state), observations = jax.lax.scan(body_fun, (init_obs, init_state), actions)
    observations = jnp.concatenate([init_obs[None, :], observations], axis=0)

    return observations, last_state


def save_model(filename: str | pathlib.Path, hyperparams: dict, model: eqx.Module):
    """Store the given model + hyperparameters at the specified file path."""
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load_model(filename: str, model_class: eqx.Module):
    """Load the model and hyperparameters from the specified file path.

    Note that the hyperparameters must be sufficient to initialized the given
    model class.
    """
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        hyperparams["key"] = jnp.array(hyperparams["key"], dtype=jnp.uint32)
        model = model_class(**hyperparams)
        return eqx.tree_deserialise_leaves(f, model)
