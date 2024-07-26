"""Zehn zahme Ziegen ziehen zehn Zentner Zucker zum Zoo."""

import json
import datetime
import argparse
import warnings

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")

import diffrax

import exciting_environments as excenvs

from exciting_exciting_systems.related_work.np_reimpl.pendulum import Pendulum
from exciting_exciting_systems.related_work.algorithms import excite_with_sGOATS


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

    env_params = dict(batch_size=1, tau=2e-2, max_torque=5, g=9.81, l=1, m=1, env_solver=diffrax.Tsit5())
    env = excenvs.make(
        env_id="Pendulum-v0",
        batch_size=env_params["batch_size"],
        action_constraints={"torque": env_params["max_torque"]},
        static_params={"g": env_params["g"], "l": env_params["l"], "m": env_params["m"]},
        solver=env_params["env_solver"],
        tau=env_params["tau"],
    )

    alg_params = dict(
        n_amplitudes=360,
        n_amplitude_groups=12,
        reuse_observations=True,
        bounds_duration=(10, 100),
        population_size=50,
        n_generations=50,
        featurize=lambda x: x,
        compress_data=True,
        compression_target_N=500,
        compression_dist_th=0.1,
        compression_feature_dim=-2,
        rho_obs=1e3,
        rho_act=1e3,
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
        env_solver=diffrax.Tsit5(),
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
        n_amplitudes=260,
        n_amplitude_groups=13,
        reuse_observations=True,
        bounds_duration=(20, 100),
        population_size=50,
        n_generations=100,
        compress_data=True,
        compression_target_N=500,
        compression_dist_th=0.1,
        compression_feature_dim=-2,
        rho_obs=1e3,
        rho_act=1e3,
        featurize=lambda x: x,
    )
    seeds = list(np.arange(101, 201))
    ## End fluid_tank experiment parameters

### End experiment parameters #########################################################################################


### Start experiments #################################################################################################

for exp_idx, seed in enumerate(seeds):

    print("Running experiment", exp_idx, f"(seed: {seed}) on '{sys_name}'")

    exp_params = dict(
        seed=int(seed),
        alg_params=alg_params,
        env_params=env_params,
    )

    # setup PRNG
    rng = np.random.default_rng(seed=seed)

    # run excitation algorithm
    observations, actions = excite_with_sGOATS(
        n_amplitudes=alg_params["n_amplitudes"],
        n_amplitude_groups=alg_params["n_amplitude_groups"],
        reuse_observations=alg_params["reuse_observations"],
        env=env,
        bounds_duration=alg_params["bounds_duration"],
        population_size=alg_params["population_size"],
        n_generations=alg_params["n_generations"],
        featurize=alg_params["featurize"],
        compress_data=alg_params["compress_data"],
        compression_target_N=alg_params["compression_target_N"],
        compression_dist_th=alg_params["compression_dist_th"],
        compression_feat_dim=alg_params["compression_feature_dim"],
        rho_obs=alg_params["rho_obs"],
        rho_act=alg_params["rho_act"],
        rng=np.random.default_rng(seed=exp_params["seed"]),
        verbose=False,
        plot_every_subsequence=False,
    )

    observations = [obs.tolist() for obs in observations]
    actions = [act.tolist() for act in actions]

    # save parameters
    file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    with open(f"../results/sgoats/{sys_name}/params_{file_name}.json", "w") as fp:
        safe_json_dump(exp_params, fp)

    # save observations + actions
    with open(f"../results/sgoats/{sys_name}/data_{file_name}.json", "w") as fp:
        json.dump(dict(observations=observations, actions=actions), fp)

    jax.clear_caches()

### End experiments ###################################################################################################
