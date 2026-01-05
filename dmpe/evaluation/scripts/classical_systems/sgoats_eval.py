"""Zehn zahme Ziegen ziehen zehn Zentner Zucker zum Zoo."""

import json
import datetime
import argparse
import warnings
import os
import pathlib

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")

from dmpe.data_management import DataPaths
from dmpe.excitation.excitation_utils import soft_penalty
from dmpe.related_work.algorithms import excite_with_sGOATS
from dmpe.utils.env_utils.fluid_tank_utils import setup_env as setup_fluid_tank_env
from dmpe.utils.env_utils.pendulum_utils import setup_env as setup_pendulum_env
from dmpe.utils.env_utils.cart_pole_utils import setup_env as setup_cart_pole_env


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

args = parser.parse_args()
sys_name = args.sys_name


### Start experiment parameters #######################################################################################
if sys_name == "pendulum":
    ## Start pendulum experiment parameters

    env, _, _, env_params = setup_pendulum_env()

    penalty_function = lambda x, u: 1e3 * soft_penalty(a=x, a_max=1, penalty_order=2) + 1e3 * soft_penalty(
        a=u, a_max=1, penalty_order=2
    )

    alg_params = dict(
        n_amplitudes=360,
        n_amplitude_groups=36,
        reuse_observations=True,
        bounds_duration=(10, 100),
        population_size=50,
        n_generations=25,
        featurize=lambda x: x,
        compress_data=True,
        compression_target_N=500,
        compression_dist_th=0.1,
        compression_feature_dim=-2,
        penalty_function=penalty_function,
    )
    seeds = list(np.arange(101, 131))
    ## End pendulum experiment parameters

elif sys_name == "fluid_tank":
    ## Start fluid_tank experiment parameters

    env, _, _, env_params = setup_fluid_tank_env()

    penalty_function = lambda x, u: 1e3 * soft_penalty(a=x, a_max=1, penalty_order=2) + 1e3 * soft_penalty(
        a=u, a_max=1, penalty_order=2
    )

    alg_params = dict(
        n_amplitudes=779,
        n_amplitude_groups=41,
        reuse_observations=True,
        bounds_duration=(5, 50),
        population_size=50,
        n_generations=25,
        compress_data=True,
        compression_target_N=500,
        compression_dist_th=0.1,
        compression_feature_dim=-2,
        penalty_function=penalty_function,
        featurize=lambda x: x,
    )
    seeds = list(np.arange(101, 131))
    ## End fluid_tank experiment parameters

elif sys_name == "cart_pole":
    ## Start cart_pole experiment parameters

    env, _, _, env_params = setup_cart_pole_env()

    penalty_function = lambda x, u: 1e3 * soft_penalty(a=x, a_max=1, penalty_order=2) + 1e3 * soft_penalty(
        a=u, a_max=1, penalty_order=2
    )

    alg_params = dict(
        n_amplitudes=720,
        n_amplitude_groups=72,
        reuse_observations=True,
        bounds_duration=(1, 100),
        population_size=50,
        n_generations=25,
        featurize=lambda x: x,
        compress_data=True,
        compression_feature_dim=-2,
        compression_target_N=500,
        compression_dist_th=0.1,
        penalty_function=penalty_function,
        penalty_order=2,
    )

    seeds = list(np.arange(101, 131))

    ## End cart_pole experiment parameters

### End experiment parameters #########################################################################################


### Start experiments #################################################################################################

for exp_idx, seed in enumerate(seeds):

    print("Running experiment", exp_idx, f"(seed: {seed}) on '{sys_name}'")

    results_path = DataPaths().se_cs_experiments / pathlib.Path("sgoats") / pathlib.Path(sys_name)
    print(f"Results will be written to: '{results_path}'.")
    assert results_path.exists(), (
        f"The expected results path '{results_path}' does not seem to exist. Please create the necessary file structure "
        + "or adapt the path."
    )

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
        penalty_function=alg_params["penalty_function"],
        rng=np.random.default_rng(seed=exp_params["seed"]),
        verbose=False,
        plot_every_subsequence=False,
    )

    observations = [obs.tolist() for obs in observations]
    actions = [act.tolist() for act in actions]

    # save parameters
    file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(results_path / pathlib.Path(f"params_{file_name}.json"), "w") as fp:
        safe_json_dump(exp_params, fp)

    # save observations + actions
    with open(results_path / pathlib.Path(f"data_{file_name}.json"), "w") as fp:
        json.dump(dict(observations=observations, actions=actions), fp)

    jax.clear_caches()

### End experiments ###################################################################################################
