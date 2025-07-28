"""Script for training models based on provided experiment data."""

import argparse
import os
import pathlib
import datetime

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


import jax
import jax.numpy as jnp
import optax
import equinox as eqx

import exciting_environments as excenvs
from dmpe.data_management import DataPaths
from dmpe.utils.env_utils.fluid_tank_utils import setup_env as setup_fluid_tank_env
from dmpe.utils.env_utils.pendulum_utils import setup_env as setup_pendulum_env
from dmpe.utils.env_utils.cart_pole_utils import setup_env as setup_cart_pole_env
from dmpe.models.models import NeuralEulerODE, NeuralEulerODEPendulum, NeuralEulerODECartpole
from dmpe.evaluation.experiment_utils import load_experiment_results, get_experiment_ids
from dmpe.evaluation.model_evaluation import ModelEvaluator, NodeModelWrapper, EnvWrapper
from dmpe.evaluation.data_evaluation import DataEvaluator
from dmpe.evaluation.utils import default_constraint_function
from dmpe.evaluation.exp_data_model_learning import train_model_on_experiment_data, ModelExpDataResult

from params import get_experiment_params


def main(
    env: excenvs.CoreEnvironment,
    model_class: eqx.Module,
    featurize: callable,
    model_params: dict,
    model_trainer_params: dict,
    data_in_path: pathlib.Path,
    data_out_path: pathlib.Path,
    n_datapoints: int,
    data_start: int = 0,
):
    # setup parameters (TODO: Should these be done with a script specifically for a given env)
    points_per_dim = 20  # grid for model eval (TODO: potentially replace with LHS)
    n_iters = 100

    seeds = jnp.arange(0, 10, 1).tolist()

    # setup all necessary objects
    wrapped_env = EnvWrapper(env, featurize=featurize)
    obs_dim = env.reset(env.env_properties)[0].shape[-1]
    act_dim = env.action_dim

    model_evaluator = ModelEvaluator(
        constraint_function=default_constraint_function,
        gt_model=wrapped_env,
        obs_dim=obs_dim,
        act_dim=act_dim,
        validation_points_per_dim=points_per_dim,
        tau=env.tau,
    )

    data_evaluator = DataEvaluator(
        constraint_function=default_constraint_function,
        data_dim=obs_dim + act_dim,
        points_per_dim=points_per_dim,
    )

    # get all experiment ids that were specified (I guess put all relevant experiments in an extra folder?)
    relevant_experiment_ids = get_experiment_ids(data_in_path)
    for experiment_idx, exp_id in enumerate(relevant_experiment_ids):
        print("Experiment ID:", exp_id)
        _, observations, actions, _ = load_experiment_results(
            exp_id,
            data_in_path,
            model_class=None,  # internal excitation model is unused anyways
        )

        # TODO: reduce_dataset? start: start + n_datapoints
        assert data_start + n_datapoints <= observations.shape[0]
        assert data_start + n_datapoints <= actions.shape[0]

        observations = observations[data_start : data_start + n_datapoints]
        actions = actions[data_start : data_start + n_datapoints]

        trained_models = []
        model_errors = []

        for _, seed in enumerate(seeds):
            trained_model, model_errors_dmpe = train_model_on_experiment_data(
                key=jax.random.key(seed),
                observations=observations,
                actions=actions,
                model_trainer_params=model_trainer_params,
                model_params=model_params,
                n_iters=n_iters,
                model_class=model_class,
                model_evaluator=model_evaluator,
            )

            trained_models.append(trained_model)
            model_errors.append(model_errors_dmpe)

        result = ModelExpDataResult.from_data(
            exp_id=exp_id,
            seeds=seeds,
            observations=observations,
            actions=actions,
            data_jsd=data_evaluator.get_metrics(data_points=jnp.concatenate([observations, actions], axis=-1))["jsd"],
            model_params=model_params,
            model_class=model_class,
            models=trained_models,
            model_errors=jnp.array(model_errors),
        )
        name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = data_out_path / f"{name}_based_on_{exp_id}.eqx"
        result.save_to_file(file_path)
        print("Successfully stored learned models and model errors.")
        jax.clear_caches()
        print(100 * "#")


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description="Train models on the provided datasets.")
    parser.add_argument(
        "--data_in_path",
        type=str,
        help=(
            "File path for the inputs relative to the data root specified"
            + " in dmpe.data_management.DataPaths().data_root."
        ),
    )  # TODO: Hardcode?
    parser.add_argument(
        "--data_out_path",
        type=str,
        help=(
            "File path for the outputs relative to the data root specified"
            + " in dmpe.data_management.DataPaths().data_root."
        ),
    )
    parser.add_argument(
        "--env_type",
        type=str,
        help="Environment to consider. One of ['fluid_tank', 'pendulum', 'cart_pole']",
    )
    parser.add_argument(
        "--training_setup",
        type=str,
        help=(
            "Chooses one of the hyperparameter presets for model training."
            + " One of ['2step', '10step_small', '10step_large', '50step_large]."
        ),
    )
    parser.add_argument(
        "--n_datapoints",
        type=int,
        help=(
            "Length of the data subset for training. The data indexed with datastart:data_start+n_datapoints is used"
            + " for training. If '-1' is used, the training is run for all length 1000 to 15000 in increments of"
            + " 1000 steps (i.e., 1000, 2000, 3000, ..., 14000, 15000)."
        ),
    )
    parser.add_argument("--data_start", type=int, default=0, help="Data index at which to start the training subset.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id to use.")
    args = parser.parse_args()

    ## setup based on specified parameters

    data_in_path = DataPaths().data_root / args.data_in_path
    data_out_path = DataPaths().data_root / args.data_out_path

    # set gpu for run
    gpus = jax.devices()
    jax.config.update("jax_default_device", gpus[args.gpu_id])
    print("Running on GPU with idx", args.gpu_id)
    print("Considering env type:", args.env_type)
    print(f"Found {len(get_experiment_ids(data_in_path))} at path {data_in_path}.")
    print(f"Writing output to {data_out_path}.")
    print(f"Using data from index {args.data_start} until {args.data_start + args.n_datapoints} for training.")

    # create corresponding env
    if args.env_type == "fluid_tank":
        env, _, featurize, _ = setup_fluid_tank_env()
        model_class = NeuralEulerODE

    elif args.env_type == "pendulum":
        env, _, featurize, _ = setup_pendulum_env()
        model_class = NeuralEulerODEPendulum

    elif args.env_type == "cart_pole":
        env, _, featurize, _ = setup_cart_pole_env()
        model_class = NeuralEulerODECartpole
    else:
        raise ValueError(f"Environment {args.env_type} could not be found.")

    model_params, model_trainer_params = get_experiment_params(
        args.training_setup,
        env,
        featurize,
    )

    # run model training
    if args.n_datapoints == -1:
        for n_datapoints in jnp.arange(1000, 15001, 1000):
            main(
                env,
                model_class,
                featurize,
                model_params,
                model_trainer_params,
                pathlib.Path(data_in_path),
                pathlib.Path(data_out_path),
                n_datapoints=n_datapoints,
                data_start=args.data_start,
            )
    else:
        main(
            env,
            model_class,
            featurize,
            model_params,
            model_trainer_params,
            pathlib.Path(data_in_path),
            pathlib.Path(data_out_path),
            n_datapoints=args.n_datapoints,
            data_start=args.data_start,
        )
