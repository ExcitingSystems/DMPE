import diffrax
import exciting_environments as excenvs
import jax.numpy as jnp

from dmpe.excitation.excitation_utils import soft_penalty


def setup_env():
    env_params = dict(
        batch_size=1,
        tau=5,
        max_height=3,
        max_inflow=0.2,
        base_area=jnp.pi,
        orifice_area=jnp.pi * 0.1**2,
        c_d=0.6,
        g=9.81,
        env_solver=diffrax.Tsit5(),
    )
    env = excenvs.make(
        "FluidTank-v0",
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

    return env, penalty_function
