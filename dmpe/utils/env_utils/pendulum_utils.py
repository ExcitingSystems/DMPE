from typing import Callable

import diffrax
import exciting_environments as excenvs
import jax.numpy as jnp

from dmpe.excitation.excitation_utils import soft_penalty


def setup_env() -> tuple[excenvs.CartPole, Callable, Callable, dict]:
    """Setup the pendulum environment and utilities.

    Args:
        -

    Returns:
        env (excenvs.CartPole): The actual environment object
        penalty_function (Callable): The penalty function that incorporates the constraints of the system
        featurize (Callable): The featurize function mainly used for wrapping angles
        env_params (dict): The parameters used to initialize the environment
    """
    env_params = dict(batch_size=None, tau=2e-2, max_torque=5, g=9.81, l=1, m=1, env_solver=diffrax.Tsit5())
    env = excenvs.EnvironmentRegistry.PENDULUM.make(
        batch_size=env_params["batch_size"],
        action_normalizations={
            "torque": excenvs.utils.MinMaxNormalization(min=-env_params["max_torque"], max=env_params["max_torque"])
        },
        static_params={"g": env_params["g"], "l": env_params["l"], "m": env_params["m"]},
        solver=env_params["env_solver"],
        tau=env_params["tau"],
    )

    penalty_function = lambda x, u: soft_penalty(a=x, a_max=1, penalty_order=2) + soft_penalty(
        a=u, a_max=1, penalty_order=2
    )

    def featurize(obs):
        """The angle itself is difficult to properly interpret in the loss as angles
        such as 1.99 * pi and 0 are essentially the same. Therefore the angle is
        transformed to sin(phi) and cos(phi) for comparison in the loss."""
        feat_obs = jnp.stack([jnp.sin(obs[..., 0] * jnp.pi), jnp.cos(obs[..., 0] * jnp.pi), obs[..., 1]], axis=-1)
        return feat_obs

    return env, penalty_function, featurize, env_params
