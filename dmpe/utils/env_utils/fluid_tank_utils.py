from typing import Callable

import diffrax
import exciting_environments as excenvs
import jax.numpy as jnp

from dmpe.excitation.excitation_utils import soft_penalty


def setup_env() -> tuple[excenvs.CartPole, Callable, Callable, dict]:
    """Setup the fluid tank environment and utilities.

    Args:
        -

    Returns:
        env (excenvs.CartPole): The actual environment object
        penalty_function (Callable): The penalty function that incorporates the constraints of the system
        featurize (Callable): The featurize function (identity for the fluid tank)
        env_params (dict): The parameters used to initialize the environment
    """
    env_params = dict(
        batch_size=None,
        tau=5,
        max_height=3,
        max_inflow=0.2,
        base_area=jnp.pi,
        orifice_area=jnp.pi * 0.1**2,
        c_d=0.6,
        g=9.81,
        env_solver=diffrax.Tsit5(),
    )
    env = excenvs.EnvironmentRegistry.FLUID_TANK.make(
        physical_normalizations=dict(height=excenvs.utils.MinMaxNormalization(min=0, max=env_params["max_height"])),
        action_normalizations=dict(inflow=excenvs.utils.MinMaxNormalization(min=0, max=env_params["max_inflow"])),
        static_params=dict(
            base_area=env_params["base_area"],
            orifice_area=env_params["orifice_area"],
            c_d=env_params["c_d"],
            g=env_params["g"],
        ),
        tau=env_params["tau"],
        solver=env_params["env_solver"],
    )
    penalty_function = lambda x, u: soft_penalty(a=x, a_max=1, penalty_order=2) + soft_penalty(
        a=u, a_max=1, penalty_order=2
    )

    featurize = lambda x: x

    return env, penalty_function, featurize, env_params
