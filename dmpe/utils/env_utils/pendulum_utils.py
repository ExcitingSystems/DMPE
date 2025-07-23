import diffrax
import exciting_environments as excenvs
import jax.numpy as jnp

from dmpe.excitation.excitation_utils import soft_penalty


def setup_env() -> tuple[excenvs.Pendulum, callable, dict]:
    env_params = dict(batch_size=1, tau=2e-2, max_torque=5, g=9.81, l=1, m=1, env_solver=diffrax.Tsit5())
    env = excenvs.make(
        env_id="Pendulum-v0",
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
    return env, penalty_function, env_params
