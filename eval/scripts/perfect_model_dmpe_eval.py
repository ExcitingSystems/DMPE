import json
import datetime
import os
import argparse

import numpy as np
import jax
import jax.numpy as jnp

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
gpus = jax.devices()
jax.config.update("jax_default_device", gpus[0])

import diffrax
from haiku import PRNGSequence

import exciting_environments as excenvs

from exciting_exciting_systems.utils.signals import aprbs
from exciting_exciting_systems.algorithms import excite_with_dmpe
from exciting_exciting_systems.models.model_utils import ModelEnvWrapperFluidTank, ModelEnvWrapperPendulum


def safe_json_dump(obj, fp):
    default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
    return json.dump(obj, fp, default=default)


parser = argparse.ArgumentParser(description="Process 'sys_name' to choose the system to experiment on.")
parser.add_argument(
    "sys_name",
    metavar="sys_name",
    type=str,
    help="The name of the environment. Options are ['pendulum', 'fluid_tank'].",
)

args = parser.parse_args()
sys_name = args.sys_name

### Start experiment parameters #######################################################################################
if sys_name == "pendulum":
    ## Start pendulum experiment parameters

    env_params = dict(batch_size=1, tau=2e-2, max_torque=5, g=9.81, l=1, m=1, env_solver=diffrax.Euler())
    env = excenvs.make(
        env_id="Pendulum-v0",
        batch_size=env_params["batch_size"],
        action_constraints={"torque": env_params["max_torque"]},
        static_params={"g": env_params["g"], "l": env_params["l"], "m": env_params["m"]},
        solver=env_params["env_solver"],
        tau=env_params["tau"],
    )
    alg_params = dict(
        bandwidth=0.05,
        n_prediction_steps=50,
        points_per_dim=50,
        action_lr=1e-1,
        n_opt_steps=10,
        rho_obs=1,
        rho_act=1,
        penalty_order=2,
        clip_action=False,
    )

    exp_params = dict(
        seed=None,
        n_timesteps=15_000,
        model_class=None,
        env_params=env_params,
        alg_params=alg_params,
        model_trainer_params=None,
        model_params=None,
        model_env_wrapper=ModelEnvWrapperPendulum,
    )
    seeds = list(np.arange(101, 201))
    ## End pendulum experiment parameters

elif sys_name == "fluid_tank":
    ## Start fluid_tank experiment parameters

    env_params = dict(
        batch_size=1,
        tau=5e-1,
        max_height=3,
        max_inflow=0.2,
        base_area=jnp.pi,
        orifice_area=jnp.pi * 0.1**2,
        c_d=0.6,
        g=9.81,
        env_solver=diffrax.Euler(),
    )
    env = excenvs.make(
        "FluidTank-v0",
        physical_constraints=dict(height=env_params["max_height"]),
        action_constraints=dict(inflow=env_params["max_inflow"]),
        static_params=dict(
            base_area=env_params["base_area"],
            orifice_area=env_params["orifice_area"],
            c_d=env_params["c_d"],
            g=env_params["g"],
        ),
        tau=env_params["tau"],
        solver=env_params["env_solver"],
    )

    alg_params = dict(
        bandwidth=0.025,  # 0.05
        n_prediction_steps=100,
        points_per_dim=50,
        action_lr=1e-1,
        n_opt_steps=10,
        rho_obs=1,
        rho_act=1,
        penalty_order=1,
        clip_action=True,
    )

    exp_params = dict(
        seed=None,
        n_timesteps=15_000,
        model_class=None,
        env_params=env_params,
        alg_params=alg_params,
        model_trainer_params=None,
        model_params=None,
        model_env_wrapper=ModelEnvWrapperFluidTank,
    )
    seeds = list(np.arange(101, 201))
    ## End fluid_tank experiment parameters

### End experiment parameters #########################################################################################

for exp_idx, seed in enumerate(seeds):
    print("Running experiment", exp_idx, f"(seed: {seed}) on '{sys_name}'")
    exp_params["seed"] = int(seed)

    # setup PRNG
    key = jax.random.PRNGKey(seed=exp_params["seed"])
    data_key, _, _, expl_key, key = jax.random.split(key, 5)
    data_rng = PRNGSequence(data_key)

    # initial guess
    proposed_actions = aprbs(exp_params["alg_params"]["n_prediction_steps"], env.batch_size, 1, 10, next(data_rng))[0]

    # run excitation algorithm
    observations, actions, model, density_estimate, losses, proposed_actions = excite_with_dmpe(
        env,
        exp_params,
        proposed_actions,
        None,
        expl_key,
    )

    # save parameters
    file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    with open(f"../results/perfect_model_dmpe/{sys_name}/params_{file_name}.json", "w") as fp:
        safe_json_dump(exp_params, fp)

    # save observations + actions
    with open(f"../results/perfect_model_dmpe/{sys_name}/data_{file_name}.json", "w") as fp:
        json.dump(dict(observations=observations.tolist(), actions=actions.tolist()), fp)

    jax.clear_caches()

### End experiments ###################################################################################################
