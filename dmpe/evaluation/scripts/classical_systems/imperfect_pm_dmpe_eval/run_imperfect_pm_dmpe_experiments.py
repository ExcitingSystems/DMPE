import json
import datetime
import argparse
import os
import pathlib
from tqdm import tqdm

import numpy as np
import jax
import jax.numpy as jnp


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import optax
from haiku import PRNGSequence
import jax_dataclasses as jdc

from dmpe.data_management import DataPaths
from dmpe.utils.signals import aprbs
from dmpe.utils.density_estimation import select_bandwidth, get_uniform_target_distribution
from dmpe.algorithms.algorithms import excite_with_dmpe
from dmpe.utils.env_utils.fluid_tank_utils import setup_env as setup_fluid_tank_env
from dmpe.utils.env_utils.pendulum_utils import setup_env as setup_pendulum_env
from dmpe.utils.env_utils.cart_pole_utils import setup_env as setup_cart_pole_env

from dmpe.evaluation.imperfect_pm_dmpe import excite_with_imperfect_model


def safe_json_dump(obj, fp):
    default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
    return json.dump(obj, fp, default=default)


parser = argparse.ArgumentParser(description="Process 'sys_name' to choose the system to experiment on.")
parser.add_argument(
    "sys_name",
    metavar="sys_name",
    type=str,
    help="The name of the environment. Options are ['pendulum', 'fluid_tank', 'cart_pole'].",
)
parser.add_argument("--gpu_id", type=int, default=0, help="GPU id to use.")

args = parser.parse_args()
sys_name = args.sys_name

gpus = jax.devices()
gpu_id = args.gpu_id
jax.config.update("jax_default_device", gpus[args.gpu_id])

if sys_name == "fluid_tank":
    env, penalty_function, featurize, env_params = setup_fluid_tank_env()
    parameter_ranges = {}
    parameter_ranges = dict(
        base_area=jnp.pi * jnp.arange(0.6, 1.41, 0.2),
        orifice_area=jnp.pi * 0.1**2 * jnp.arange(0.6, 1.41, 0.2),
    )
elif sys_name == "pendulum":
    env, penalty_function, featurize, env_params = setup_pendulum_env()
    parameter_ranges = dict(
        m=jnp.arange(0.6, 1.41, 0.2),
        l=jnp.arange(0.6, 1.41, 0.2),
    )
elif sys_name == "cart_pole":
    env, penalty_function, featurize, env_params = setup_cart_pole_env()
    parameter_ranges = dict(
        m_c=jnp.arange(0.6, 1.41, 0.2),
        m_p=jnp.arange(0.1, 1.11, 0.2),
        l=jnp.arange(0.1, 1.11, 0.2),
    )
else:
    raise NotImplementedError(f"System '{sys_name}' is unknown. Choose from ['pendulum', 'fluid_tank', 'cart_pole'].")

parameter_combinations = jnp.meshgrid(*parameter_ranges.values(), indexing="ij")
n_parameters = len(parameter_combinations)
parameter_combinations = jnp.stack([_x for _x in parameter_combinations], axis=-1)
parameter_combinations = parameter_combinations.reshape(-1, n_parameters)

seeds = [125]  ##, 156, 412]
default_params = env.env_properties.static_params

results_path = DataPaths().imperfect_pm_dmpe_experiments / sys_name

for parameters in tqdm(parameter_combinations):

    print(parameters)

    if sys_name == "fluid_tank":
        A, A_o = tuple(parameters)
        new_static_params = env.StaticParams(
            base_area=A.item(),
            orifice_area=A_o.item(),
            c_d=default_params.c_d,
            g=default_params.g,
        )
    elif sys_name == "pendulum":
        m, l = tuple(parameters)
        new_static_params = env.StaticParams(
            l=l.item(),
            m=m.item(),
            g=default_params.g,
        )
    elif sys_name == "cart_pole":
        m_c, m_p, l = tuple(parameters)
        new_static_params = env.StaticParams(
            l=l.item(),
            m_p=m_p.item(),
            m_c=m_c.item(),
            g=default_params.g,
            mu_p=default_params.mu_p,
            mu_c=default_params.mu_c,
        )

    for seed in seeds:
        observations, actions, model, exp_params, debug_out = excite_with_imperfect_model(
            seed=seed,
            n_time_steps=5000,
            env=env,
            penalty_function=penalty_function,
            env_params=env_params,
            static_params=new_static_params,
        )

        exp_params["model_params"]["new_static_params"] = jdc.asdict(exp_params["model_params"]["new_static_params"])

        # save parameters
        file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(results_path / pathlib.Path(f"params_{file_name}.json"), "w") as fp:
            safe_json_dump(exp_params, fp)

        # save observations + actions
        with open(results_path / pathlib.Path(f"data_{file_name}.json"), "w") as fp:
            json.dump(dict(observations=observations.tolist(), actions=actions.tolist()), fp)

        jax.clear_caches()
