import diffrax
import exciting_environments as excenvs
import jax.numpy as jnp

from dmpe.excitation.excitation_utils import soft_penalty


def setup_env() -> tuple[excenvs.CartPole, callable]:
    env_params = dict(
        batch_size=1,
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
    env = excenvs.make(
        env_id="CartPole-v0",
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

    return env, penalty_function
