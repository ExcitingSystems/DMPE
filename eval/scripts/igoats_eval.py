import json
import datetime
import argparse

import numpy as np
import jax
import jax.numpy as jnp
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

jax.config.update("jax_platform_name", "cpu")

import diffrax

import exciting_environments as excenvs

from exciting_exciting_systems.related_work.algorithms import excite_with_iGOATS


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

    h = 10
    a = 10

    alg_params = dict(
        prediction_horizon=h,
        application_horizon=a,
        bounds_amplitude=(-1, 1),
        bounds_duration=(10, 100),
        population_size=50,
        n_generations=25,
        featurize=lambda x: x,
        rng=None,
        compress_data=True,
        compression_target_N=500,
        rho_obs=1e3,
        rho_act=1e3,
        compression_feat_dim=-2,
        compression_dist_th=0.1,
        penalty_order=2,
    )
    seeds = list(np.arange(101, 131))
    ## End pendulum experiment parameters

elif sys_name == "fluid_tank":
    ## Start pendulum experiment parameters

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

    h = 10
    a = 10

    alg_params = dict(
        prediction_horizon=h,
        application_horizon=a,
        bounds_amplitude=(-1, 1),
        bounds_duration=(5, 50),
        population_size=50,
        n_generations=25,
        featurize=lambda x: x,
        rng=None,
        compress_data=True,
        compression_target_N=500,
        rho_obs=1e3,
        rho_act=1e3,
        compression_feat_dim=-2,
        compression_dist_th=0.1,
        penalty_order=2,
    )

    seeds = list(np.arange(101, 131))
    ## End fluid_tank experiment parameters

elif sys_name == "cart_pole":
    ## Start cart_pole experiment parameters

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
        physical_constraints={
            "deflection": 2.4,
            "velocity": 8,
            "theta": jnp.pi,
            "omega": 8,
        },
        env_solver=diffrax.Tsit5(),
    )
    env = excenvs.make(
        env_id="CartPole-v0",
        batch_size=env_params["batch_size"],
        action_constraints={"force": env_params["max_force"]},
        physical_constraints=env_params["physical_constraints"],
        static_params=env_params["static_params"],
        solver=env_params["env_solver"],
        tau=env_params["tau"],
    )

    h = 10
    a = 5  # to help with stabilization?

    alg_params = dict(
        prediction_horizon=h,
        application_horizon=a,
        bounds_amplitude=(-1, 1),
        bounds_duration=(1, 100),
        population_size=50,
        n_generations=25,
        featurize=lambda x: x,
        rng=None,
        compress_data=True,
        compression_target_N=500,
        rho_obs=1e3,
        rho_act=1e3,
        penalty_order=2,
        compression_feat_dim=-2,
        compression_dist_th=0.1,
    )

    seeds = list(np.arange(101, 131))

    ## End cart_pole experiment parameters

### End experiment parameters #########################################################################################


### Start experiments #################################################################################################

for exp_idx, seed in enumerate(seeds):

    print("Running experiment", exp_idx, f"(seed: {seed}) on '{sys_name}'")

    exp_params = dict(
        n_timesteps=15000,
        seed=int(seed),
        alg_params=alg_params,
        env_params=env_params,
    )

    # run excitation algorithm
    observations, actions = excite_with_iGOATS(
        n_timesteps=exp_params["n_timesteps"],
        env=env,
        prediction_horizon=alg_params["prediction_horizon"],
        application_horizon=alg_params["application_horizon"],
        bounds_amplitude=alg_params["bounds_amplitude"],
        bounds_duration=alg_params["bounds_duration"],
        population_size=alg_params["population_size"],
        n_generations=alg_params["n_generations"],
        featurize=alg_params["featurize"],
        rng=np.random.default_rng(seed),
        compress_data=alg_params["compress_data"],
        compression_target_N=alg_params["compression_target_N"],
        rho_obs=alg_params["rho_obs"],
        rho_act=alg_params["rho_act"],
        penalty_order=alg_params["penalty_order"],
        compression_feat_dim=alg_params["compression_feat_dim"],
        compression_dist_th=alg_params["compression_dist_th"],
        plot_subsequences=False,
    )

    observations = [obs.tolist() for obs in observations]
    actions = [act.tolist() for act in actions]

    # save parameters
    file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    with open(f"../results/igoats/{sys_name}/params_{file_name}.json", "w") as fp:
        safe_json_dump(exp_params, fp)

    # save observations + actions
    with open(f"../results/igoats/{sys_name}/data_{file_name}.json", "w") as fp:
        json.dump(dict(observations=observations, actions=actions), fp)

    jax.clear_caches()

### End experiments ###################################################################################################
