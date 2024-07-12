"""Zehn zahme Ziegen ziehen zehn Zentner Zucker zum Zoo."""

import json
import datetime
import argparse

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

    env_params = dict(batch_size=1, tau=2e-2, max_torque=8, g=9.81, l=1, m=1)
    env = Pendulum(
        batch_size=env_params["batch_size"],
        max_torque=env_params["max_torque"],
        g=env_params["g"],
        l=env_params["l"],
        m=env_params["m"],
        tau=env_params["tau"],
    )

    alg_params = dict(
        n_amplitudes=600,
        n_amplitude_groups=12,
        reuse_observations=True,
        bounds_duration=(1, 50),
        population_size=50,
        n_generations=50,
        featurize=lambda x: x,
    )
    seeds = list(np.arange(1, 101))
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
        n_amplitudes=200,
        n_amplitude_groups=10,
        reuse_observations=True,
        bounds_duration=(1, 50),
        population_size=50,
        n_generations=100,
        featurize=lambda x: x,
    )
    seeds = list(np.arange(1, 101))
    ## End fluid_tank experiment parameters

### End Experiment parameters

for seed in seeds:
    exp_params = dict(
        seed=seed,
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
        rng=np.random.default_rng(seed=exp_params["seed"]),
        verbose=True,
        plot_every_subsequence=False,
    )

    # save parameters
    file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    with open(f"../results/sgoats/params_{file_name}.json", "w") as fp:
        safe_json_dump(exp_params, fp)

    # save observations + actions
    with open(f"../results/sgoats/data_{file_name}.json", "w") as fp:
        json.dump(dict(observations=observations.tolist(), actions=actions.tolist()), fp)

    jax.clear_caches()
