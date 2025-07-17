"""Script for training models based on provided experiment data."""

import argparse
import os
import pathlib

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


def main(
    env: excenvs.CoreEnvironment,
    model_class: eqx.Module,
    featurize: callable,
    data_in_path: pathlib.Path,
    data_out_path: pathlib.Path,
):
    # setup parameters (TODO: Should these be done with a script specifically for a given env)
    points_per_dim = 20  # grid for model eval (TODO: potentially replace with LHS)

    lr = 1e-4
    n_iters = 100
    n_datapoints = 1_000  # TODO: Do I really want to do it like this?

    seeds = jnp.arange(0, 10, 1).tolist()

    # setup all necessary objects
    wrapped_env = EnvWrapper(env)
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

    model_params = dict(
        obs_dim=env.reset(env.env_properties)[0].shape[0],
        action_dim=env.action_dim,
        width_size=64,
        depth=2,
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

        # TODO: reduce_dataset?
        observations = observations[:n_datapoints]
        actions = actions[:n_datapoints]

        trained_models = []
        model_errors = []

        for _, seed in enumerate(seeds):
            trained_model, model_errors_dmpe = train_model_on_experiment_data(
                key=jax.random.key(seed),
                observations=observations,
                actions=actions,
                model_trainer_params=dict(
                    start_learning=None,
                    training_batch_size=128,
                    n_train_steps=1_000,
                    sequence_length=10,
                    featurize=featurize,
                    model_optimizer=optax.adabelief(lr),
                    tau=env.tau,
                ),
                model_params=model_params,
                n_iters=n_iters,
                model_class=model_class,
                model_evaluator=model_evaluator,
            )

            trained_models.append(trained_model)
            model_errors.append(model_errors_dmpe)

        result = ModelExpDataResult(
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
        file_path = data_out_path / f"test_{experiment_idx}.eqx"
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
            + "in dmpe.data_management.DataPaths().data_root."
        ),
    )  # TODO: Hardcode?
    parser.add_argument(
        "--data_out_path",
        type=str,
        help=(
            "File path for the outputs relative to the data root specified"
            + "in dmpe.data_management.DataPaths().data_root."
        ),
    )
    parser.add_argument(
        "--env_type",
        type=str,
        help="Environment to consider. One of ['fluid_tank', 'pendulum', 'cart_pole']",
    )
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

    # create corresponding env
    if args.env_type == "fluid_tank":
        env, _ = setup_fluid_tank_env()
        model_class = NeuralEulerODE
        featurize = lambda x: x

    elif args.env_type == "pendulum":
        env, _ = setup_pendulum_env()
        model_class = NeuralEulerODEPendulum

        def featurize(obs):
            feat_obs = jnp.stack(
                [jnp.sin(obs[..., 0] * jnp.pi), jnp.cos(obs[..., 0] * jnp.pi), obs[..., 1]],
                axis=-1,
            )
            return feat_obs

    elif args.env_type == "cart_pole":
        env, _ = setup_cart_pole_env()
        model_class = NeuralEulerODECartpole

        def featurize(obs):
            feat_obs = jnp.stack(
                [obs[..., 0], obs[..., 1], jnp.sin(obs[..., 2] * jnp.pi), jnp.cos(obs[..., 2] * jnp.pi), obs[..., 3]],
                axis=-1,
            )
            return feat_obs

    # run model training
    main(env, model_class, featurize, pathlib.Path(data_in_path), pathlib.Path(data_out_path))
