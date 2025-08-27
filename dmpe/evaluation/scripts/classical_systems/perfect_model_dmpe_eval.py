import json
import datetime
import argparse
import os
import pathlib

import numpy as np
import jax
import jax.numpy as jnp

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import optax
from haiku import PRNGSequence

from dmpe.data_management import DataPaths
from dmpe.utils.signals import aprbs
from dmpe.utils.density_estimation import select_bandwidth, get_uniform_target_distribution
from dmpe.algorithms.algorithms import excite_with_dmpe
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
parser.add_argument("--gpu_id", type=int, default=0, help="GPU id to use.")

args = parser.parse_args()
sys_name = args.sys_name

gpus = jax.devices()
gpu_id = args.gpu_id
jax.config.update("jax_default_device", gpus[args.gpu_id])

### Start experiment parameters #######################################################################################
if sys_name == "pendulum":
    ## Start pendulum experiment parameters

    env, penalty_function, _, env_params = setup_pendulum_env()
    alg_params = dict(
        bandwidth=None,
        n_prediction_steps=20,
        points_per_dim=21,
        grid_extend=1.05,
        excitation_optimizer=optax.adabelief(1e-1),
        n_opt_steps=10,
        start_optimizing=5,
        consider_action_distribution=True,
        penalty_function=penalty_function,
        target_distribution=None,
        clip_action=True,
        n_starts=5,
        reuse_proposed_actions=True,
    )

    alg_params["bandwidth"] = float(
        select_bandwidth(
            delta_z=2,
            dim=env.physical_state_dim + env.action_dim,
            n_g=alg_params["points_per_dim"],
            percentage=0.3,
        )
    )

    # overwrite target distribution
    alg_params["target_distribution"] = get_uniform_target_distribution(
        dim=3 if alg_params["consider_action_distribution"] else 2,
        points_per_dim=alg_params["points_per_dim"],
        bandwidth=alg_params["bandwidth"],
        grid_extend=alg_params["grid_extend"],
        consider_action_distribution=alg_params["consider_action_distribution"],
        penalty_function=alg_params["penalty_function"],
    )

    exp_params = dict(
        seed=None,
        n_time_steps=15_000,
        model_class=None,
        env_params=env_params,
        alg_params=alg_params,
        model_trainer_params=None,
        model_params=None,
    )
    seeds = list(np.arange(101, 131))
    ## End pendulum experiment parameters

elif sys_name == "fluid_tank":
    ## Start fluid_tank experiment parameters

    env, penalty_function, _, env_params = setup_fluid_tank_env()

    alg_params = dict(
        bandwidth=None,
        n_prediction_steps=10,
        points_per_dim=50,
        grid_extend=1.05,
        excitation_optimizer=optax.adabelief(1e-1),
        n_opt_steps=10,
        start_optimizing=5,
        consider_action_distribution=True,
        penalty_function=penalty_function,
        target_distribution=None,
        clip_action=True,
        n_starts=5,
        reuse_proposed_actions=True,
    )

    alg_params["bandwidth"] = float(
        select_bandwidth(
            delta_z=2,
            dim=env.physical_state_dim + env.action_dim,
            n_g=alg_params["points_per_dim"],
            percentage=0.3,
        )
    )

    # overwrite and target distribution
    alg_params["target_distribution"] = get_uniform_target_distribution(
        dim=2 if alg_params["consider_action_distribution"] else 1,
        points_per_dim=alg_params["points_per_dim"],
        bandwidth=alg_params["bandwidth"],
        grid_extend=alg_params["grid_extend"],
        consider_action_distribution=alg_params["consider_action_distribution"],
        penalty_function=alg_params["penalty_function"],
    )

    exp_params = dict(
        seed=None,
        n_time_steps=15_000,
        model_class=None,
        env_params=env_params,
        alg_params=alg_params,
        model_trainer_params=None,
        model_params=None,
    )
    seeds = list(np.arange(101, 131))
    ## End fluid_tank experiment parameters

elif sys_name == "cart_pole":
    ## Start cart_pole experiment parameters

    env, penalty_function, _, env_params = setup_cart_pole_env()

    alg_params = dict(
        bandwidth=0.12,
        n_prediction_steps=50,
        points_per_dim=10,
        grid_extend=1.05,
        excitation_optimizer=optax.adabelief(1e-1),
        n_opt_steps=5,
        start_optimizing=5,
        consider_action_distribution=True,
        penalty_function=penalty_function,
        target_distribution=None,
        clip_action=True,
        n_starts=5,
        reuse_proposed_actions=True,
    )

    # alg_params["bandwidth"] = float(
    #     select_bandwidth(
    #         delta_z=2,
    #         dim=env.physical_state_dim + env.action_dim,
    #         n_g=alg_params["points_per_dim"],
    #         percentage=0.1,
    #     )
    # )

    # overwrite target distribution
    alg_params["target_distribution"] = get_uniform_target_distribution(
        dim=5 if alg_params["consider_action_distribution"] else 4,
        points_per_dim=alg_params["points_per_dim"],
        bandwidth=alg_params["bandwidth"],
        grid_extend=alg_params["grid_extend"],
        consider_action_distribution=alg_params["consider_action_distribution"],
        penalty_function=alg_params["penalty_function"],
    )

    exp_params = dict(
        seed=None,
        n_time_steps=15_000,
        model_class=None,
        env_params=env_params,
        alg_params=alg_params,
        model_trainer_params=None,
        model_params=None,
    )
    seeds = list(np.arange(101, 131))

    ## End cart_pole experiment parameters

else:
    raise NotImplementedError(f"System '{sys_name}' is unknown. Choose from ['pendulum', 'fluid_tank', 'cart_pole'].")

### End experiment parameters #########################################################################################

for exp_idx, seed in enumerate(seeds):
    print("Running experiment", exp_idx, f"(seed: {seed}) on '{sys_name}'")
    exp_params["seed"] = int(seed)

    # Check that the targeted data folder actually exist:
    results_path = DataPaths().se_cs_experiments / pathlib.Path("perfect_model_dmpe") / pathlib.Path(sys_name)
    print(f"Results will be written to: '{results_path}'.")
    assert results_path.exists(), (
        f"The expected results path '{results_path}' does not seem to exist. Please create the necessary file structure "
        + "or adapt the path."
    )

    # setup PRNG
    key = jax.random.PRNGKey(seed=exp_params["seed"])
    data_key, _, _, expl_key, key = jax.random.split(key, 5)
    data_rng = PRNGSequence(data_key)

    # initial guess
    proposed_actions = jnp.hstack(
        [
            aprbs(alg_params["n_prediction_steps"], env.batch_size, 1, 10, next(data_rng))[0]
            for _ in range(env.action_dim)
        ]
    )
    # run excitation algorithm
    observations, actions, model, density_estimate, losses, proposed_actions, _ = excite_with_dmpe(
        env,
        exp_params,
        proposed_actions,
        None,
        expl_key,
    )

    # save parameters
    file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(results_path / pathlib.Path(f"params_{file_name}.json"), "w") as fp:
        safe_json_dump(exp_params, fp)

    # save observations + actions
    with open(results_path / pathlib.Path(f"data_{file_name}.json"), "w") as fp:
        json.dump(dict(observations=observations.tolist(), actions=actions.tolist()), fp)

    jax.clear_caches()

### End experiments ###################################################################################################
