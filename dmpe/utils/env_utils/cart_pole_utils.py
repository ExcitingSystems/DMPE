from typing import Callable

import diffrax
import exciting_environments as excenvs
import jax.numpy as jnp

from dmpe.excitation.excitation_utils import soft_penalty


def setup_env() -> tuple[excenvs.CartPole, Callable, Callable, dict]:
    """Setup the cart pole environment and utilities.

    Args:
        -

    Returns:
        env (excenvs.CartPole): The actual environment object
        penalty_function (Callable): The penalty function that incorporates the constraints of the system
        featurize (Callable): The featurize function mainly used for wrapping angles
        env_params (dict): The parameters used to initialize the environment
    """
    env_params = dict(
        batch_size=None,
        tau=2e-2,
        max_force=10,
        static_params={
            "mu_p": 0.002,
            "mu_c": 0.5,
            "l": 0.5,
            "m_p": 0.1,
            "m_c": 1,
            "g": 9.81,
        },
        physical_normalizations={
            "deflection": excenvs.utils.MinMaxNormalization(min=-2.4, max=2.4),
            "velocity": excenvs.utils.MinMaxNormalization(min=-8, max=8),
            "theta": excenvs.utils.MinMaxNormalization(min=-jnp.pi, max=jnp.pi),
            "omega": excenvs.utils.MinMaxNormalization(min=-8, max=8),
        },
        env_solver=diffrax.Tsit5(),
    )
    env = excenvs.EnvironmentRegistry.CART_POLE.make(
        batch_size=env_params["batch_size"],
        action_normalizations={
            "force": excenvs.utils.MinMaxNormalization(min=-env_params["max_force"], max=env_params["max_force"])
        },
        physical_normalizations=env_params["physical_normalizations"],
        static_params=env_params["static_params"],
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
        feat_obs = jnp.stack(
            [obs[..., 0], obs[..., 1], jnp.sin(obs[..., 2] * jnp.pi), jnp.cos(obs[..., 2] * jnp.pi), obs[..., 3]],
            axis=-1,
        )
        return feat_obs

    return env, penalty_function, featurize, env_params
