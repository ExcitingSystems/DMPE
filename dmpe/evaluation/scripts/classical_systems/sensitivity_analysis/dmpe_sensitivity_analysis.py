import json
import argparse
import datetime
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
from dmpe.models.models import NeuralEulerODEPendulum, NeuralEulerODE, NeuralEulerODECartpole
from dmpe.models.model_utils import save_model
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


if sys_name == "fluid_tank":
    h_range = np.arange(0.005, 0.1001, 0.001)
elif sys_name == "pendulum":
    h_range = np.arange(0.005, 0.2001, 0.005)
elif sys_name == "cart_pole":
    h_range = np.arange(0.005, 0.4001, 0.005)


for h in h_range:

    print(f"Starting experiments for {sys_name} with h={h}")

    ### Start experiment parameters #######################################################################################
    if sys_name == "pendulum":
        ## Start pendulum experiment parameters

        env, penalty_function, featurize, env_params = setup_pendulum_env()
        alg_params = dict(
            bandwidth=h,
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
            a=None,
        )

        # alg_params["bandwidth"] = float(
        #     select_bandwidth(
        #         delta_z=2 * alg_params["grid_extend"],
        #         dim=env.physical_state_dim + env.action_dim,
        #         n_g=alg_params["points_per_dim"],
        #         percentage=a,
        #     )
        # )

        alg_params["target_distribution"] = get_uniform_target_distribution(
            dim=3 if alg_params["consider_action_distribution"] else 2,
            points_per_dim=alg_params["points_per_dim"],
            bandwidth=alg_params["bandwidth"],
            grid_extend=alg_params["grid_extend"],
            consider_action_distribution=alg_params["consider_action_distribution"],
            penalty_function=alg_params["penalty_function"],
            act_dim=1,
            obs_dim=2,
        )

        model_trainer_params = dict(
            start_learning=alg_params["n_prediction_steps"],
            training_batch_size=128,
            n_train_steps=1,
            sequence_length=alg_params["n_prediction_steps"],
            featurize=featurize,
            model_lr=1e-4,
        )
        model_params = dict(
            obs_dim=env.physical_state_dim,
            action_dim=env.action_dim,
            width_size=128,
            depth=3,
            key=None,
        )

        exp_params = dict(
            seed=None,
            n_time_steps=15_000,
            model_class=NeuralEulerODEPendulum,
            env_params=env_params,
            alg_params=alg_params,
            model_trainer_params=model_trainer_params,
            model_params=model_params,
        )
        seeds = list(np.arange(201, 206))
        ## End pendulum experiment parameters

    elif sys_name == "fluid_tank":
        ## Start fluid_tank experiment parameters

        env, penalty_function, featurize, env_params = setup_fluid_tank_env()

        alg_params = dict(
            bandwidth=h,
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
            a=None,
        )

        # alg_params["bandwidth"] = float(
        #     select_bandwidth(
        #         delta_z=2 * alg_params["grid_extend"],
        #         dim=env.physical_state_dim + env.action_dim,
        #         n_g=alg_params["points_per_dim"],
        #         percentage=a,
        #     )
        # )

        # overwrite penalty function and target distribution
        alg_params["target_distribution"] = get_uniform_target_distribution(
            dim=2 if alg_params["consider_action_distribution"] else 1,
            points_per_dim=alg_params["points_per_dim"],
            bandwidth=alg_params["bandwidth"],
            grid_extend=alg_params["grid_extend"],
            consider_action_distribution=alg_params["consider_action_distribution"],
            penalty_function=alg_params["penalty_function"],
            act_dim=1,
            obs_dim=1,
        )

        model_trainer_params = dict(
            start_learning=alg_params["n_prediction_steps"],
            training_batch_size=128,
            n_train_steps=1,
            sequence_length=alg_params["n_prediction_steps"],
            featurize=featurize,
            model_lr=1e-4,
        )
        model_params = dict(
            obs_dim=env.physical_state_dim, action_dim=env.action_dim, width_size=128, depth=3, key=None
        )

        exp_params = dict(
            seed=None,
            n_time_steps=15_000,
            model_class=NeuralEulerODE,
            env_params=env_params,
            alg_params=alg_params,
            model_trainer_params=model_trainer_params,
            model_params=model_params,
        )
        seeds = list(np.arange(201, 206))
        ## End fluid_tank experiment parameters

    elif sys_name == "cart_pole":
        ## Start cart_pole experiment parameters

        env, penalty_function, featurize, env_params = setup_cart_pole_env()
        alg_params = dict(
            bandwidth=h,
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
            a=None,
        )

        # alg_params["bandwidth"] = float(
        #     select_bandwidth(
        #         delta_z=2 * alg_params["grid_extend"],
        #         dim=env.physical_state_dim + env.action_dim,
        #         n_g=alg_params["points_per_dim"],
        #         percentage=a,
        #     )
        # )

        # overwrite penalty function and target distribution
        alg_params["target_distribution"] = get_uniform_target_distribution(
            dim=5 if alg_params["consider_action_distribution"] else 4,
            points_per_dim=alg_params["points_per_dim"],
            bandwidth=alg_params["bandwidth"],
            grid_extend=alg_params["grid_extend"],
            consider_action_distribution=alg_params["consider_action_distribution"],
            penalty_function=alg_params["penalty_function"],
            act_dim=1,
            obs_dim=4,
        )

        model_trainer_params = dict(
            start_learning=alg_params["n_prediction_steps"],
            training_batch_size=128,
            n_train_steps=10,
            sequence_length=alg_params["n_prediction_steps"],
            featurize=featurize,
            model_lr=1e-4,
        )
        model_params = dict(
            obs_dim=env.physical_state_dim, action_dim=env.action_dim, width_size=128, depth=3, key=None
        )

        exp_params = dict(
            seed=None,
            n_time_steps=15_000,
            model_class=NeuralEulerODECartpole,
            env_params=env_params,
            alg_params=alg_params,
            model_trainer_params=model_trainer_params,
            model_params=model_params,
        )
        seeds = list(np.arange(201, 206))
        ## End cart_pole experiment parameters

    ### End experiment parameters #########################################################################################

    ### Start experiments #################################################################################################
    for exp_idx, seed in enumerate(seeds):

        print("Running experiment", exp_idx, f"(seed: {seed}) on '{sys_name}'")
        exp_params["seed"] = int(seed)

        # Check that the targeted data folder actually exist:
        results_path = DataPaths().sensitivity_analysis_experiments / pathlib.Path("dmpe") / pathlib.Path(sys_name)
        print(f"Results will be written to: '{results_path}'.")
        assert results_path.exists(), (
            f"The expected results path '{results_path}' does not seem to exist. Please create the necessary file structure "
            + "or adapt the path."
        )

        # setup PRNG
        key = jax.random.PRNGKey(seed=exp_params["seed"])
        data_key, model_key, loader_key, expl_key, key = jax.random.split(key, 5)
        data_rng = PRNGSequence(data_key)
        exp_params["model_params"]["key"] = model_key

        # initial guess
        proposed_actions = jnp.hstack(
            [
                aprbs(alg_params["n_prediction_steps"], env.batch_size, 1, 10, next(data_rng))[0]
                for _ in range(env.action_dim)
            ]
        )

        # run excitation algorithm
        observations, actions, model, density_estimate, losses, proposed_actions, _ = excite_with_dmpe(
            env, exp_params, proposed_actions, loader_key, expl_key
        )

        # save parameters
        file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(results_path / pathlib.Path(f"params_{file_name}.json"), "w") as fp:
            safe_json_dump(exp_params, fp)

        # save observations + actions
        with open(results_path / pathlib.Path(f"data_{file_name}.json"), "w") as fp:
            json.dump(dict(observations=observations.tolist(), actions=actions.tolist()), fp)

        model_params["key"] = model_params["key"].tolist()
        save_model(results_path / pathlib.Path(f"model_{file_name}.json"), hyperparams=model_params, model=model)

        jax.clear_caches()

### End experiments ###################################################################################################
