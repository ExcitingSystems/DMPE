from typing import Callable
from functools import partial
import glob
import pathlib
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import jax
import jax.numpy as jnp
import equinox as eqx

import exciting_environments as excenvs
from dmpe.related_work.random_walk import random_walk_control_law
from dmpe.evaluation.model_evaluation import RolloutComparison, EnvWrapper, NodeModelWrapper
from dmpe.evaluation.exp_data_model_learning import ModelExpDataResult
from dmpe.evaluation.metrics_utils import default_jsd


def plot_jsd_model_prediction_relation(
    data_path: pathlib.Path,
    model_class: eqx.Module,
    verbose: bool = False,
    expecting_sub_folders: bool = True,
    penalty_function: callable = None,
    recompute_jsd: bool = False,
):
    means = []
    medians = []
    jsds = []
    colors = []

    color_cycle = plt.rcParams["axes.prop_cycle"]()
    color_mapping = [next(color_cycle)["color"] for _ in range(15)]

    result_paths = (
        glob.glob(str(data_path) + "/**/*.eqx") if expecting_sub_folders else glob.glob(str(data_path) + "/*.eqx")
    )

    n_results = len(result_paths)
    print("# or results:", n_results)
    print(80 * "-")

    for result_path in tqdm(result_paths, total=len(result_paths)):
        result = ModelExpDataResult.from_file(
            filename=result_path,
            model_class=model_class,
        )
        if penalty_function is not None:
            if penalty_function(result.observations, result.actions) > 1:
                continue

        color_idx = int(result.n_datapoints / 1_000) - 1
        colors.append(color_mapping[color_idx])

        means.append(jnp.mean(jnp.array(result.model_errors), axis=0)[-1])
        medians.append(jnp.median(jnp.array(result.model_errors), axis=0)[-1])

        if recompute_jsd:
            jsd_value = default_jsd(
                result.observations,
                result.actions,
                points_per_dim=20,
                bounds=(-1, 1),
                bandwidth=0.08,  # TODO: What about this?!
                target_distribution=None,
                ca=False,
            )
        else:
            jsd_value = result.data_jsd

        jsds.append(jsd_value)

        if verbose:
            print(result.data_jsd)
            fig, _ = result.visualize()
            plt.show()
            print(80 * "-")

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    ax.scatter(
        jsds, medians, s=25, marker="x", c=colors
    )  # , c=next(colors)["color"], label=f"{data_length} data points")

    ax.set_ylabel("model prediction loss")
    ax.set_xlabel("JSD")
    # ax.set_ylabel(r"$\mathcal{L}_{\mathcal{M}}$")
    # ax.set_xlabel(r"$\mathcal{L}_{\mathrm{JSD}}$")
    ax.grid(True)
    ax.set_yscale("log")
    ax.set_xscale("log")
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="x",
            color="w",
            label=(idx + 1) * 1000,
            markerfacecolor=color_mapping[idx],
            markeredgecolor=color_mapping[idx],
            markersize=5,
            linestyle="None",
        )
        for idx in range(len(color_mapping))
    ]
    # legend_elements = [Patch(facecolor=color_mapping[idx], label=(idx + 1) * 1000) for idx in range(len(color_mapping))]
    ax.legend(handles=legend_elements, title=r"\# of datapoints")

    return fig, ax


def plot_model_rollouts(
    env: excenvs.CoreEnvironment,
    penalty_function: callable,
    featurize: callable,
    model: eqx.Module,
    batch_size: int,
    sequence_length: int,
    key: jax.random.PRNGKey,
    control_law: Callable | None = None,
):
    if control_law is None:
        control_law = partial(random_walk_control_law, n_tries=4000)

    key, init_obs_key, rollout_key = jax.random.split(key, 3)
    init_obs_keys = jax.random.split(init_obs_key, batch_size)
    init_obs, state = eqx.filter_vmap(env.reset, in_axes=(None, 0))(env.env_properties, init_obs_keys)

    wrapped_model = NodeModelWrapper(model, featurize)
    wrapped_env = EnvWrapper(env, featurize)

    rollout_comparison = RolloutComparison(
        control_law=control_law,
        penalty_function=penalty_function,
        tau=env.tau,
        env=env,
        sequence_length=sequence_length,
    )

    (env_observations, pred_gt, pred, key), metric = rollout_comparison(
        init_obs, wrapped_model, wrapped_env, key=rollout_key
    )

    feat_obs_dim = featurize(env.reset(env.env_properties)[0]).shape[-1]

    print(metric)

    for i in range(env_observations.shape[0]):
        fig, axs = plt.subplots(1, feat_obs_dim, figsize=(12, 4))

        if feat_obs_dim == 1:
            label = env.obs_description[0]

            axs.plot(pred_gt[i], label="gt_" + label)
            axs.plot(pred[i], label="pred_" + label, linestyle="--")
            axs.legend()
            axs.grid()
            axs.set_xlim(0.0, len(pred_gt[i]) - 1)
            axs.set_ylim(-1.1, 1.1)
        else:
            for ax, obs_gt, obs_pred in zip(axs, pred_gt[i].T, pred[i].T):
                ax.plot(obs_gt, label="gt_")
                ax.plot(obs_pred, label="pred_", linestyle="--")
                ax.legend()
                ax.grid()
                ax.set_xlim(0.0, len(obs_gt) - 1)
                ax.set_ylim(-1.1, 1.1)
        plt.show()


def evaluate_model_rollout(
    rollout_comparison,
    wrapped_env,
    wrapped_model,
    batch_size,
    key,
):
    env = wrapped_env.model

    key, init_obs_key, rollout_key = jax.random.split(key, 3)
    init_obs_keys = jax.random.split(init_obs_key, batch_size)

    init_obs, state = eqx.filter_vmap(env.reset, in_axes=(None, 0))(env.env_properties, init_obs_keys)
    (env_observations, pred_gt, pred, key), metric = rollout_comparison(
        init_obs, wrapped_model, wrapped_env, key=rollout_key
    )
    return metric


def plot_jsd_model_rollout_relation(
    data_path: pathlib.Path,
    env: excenvs.CoreEnvironment,
    penalty_function: callable,
    featurize: callable,
    batch_size: int,
    sequence_length: int,
    model_class: type[eqx.Module],
    key: jax.random.PRNGKey,
    control_law: Callable | None = None,
):

    if control_law is None:
        control_law = partial(random_walk_control_law, n_tries=4000)

    rollout_comparison = RolloutComparison(
        control_law=control_law,
        penalty_function=penalty_function,
        tau=env.tau,
        env=env,
        sequence_length=sequence_length,
    )
    wrapped_env = EnvWrapper(env, featurize=featurize)

    metrics = []
    jsds = []

    result_paths = glob.glob(str(data_path) + "/*.eqx")

    n_results = len(result_paths)
    print("# or results:", n_results)
    print(80 * "-")
    for result_path in tqdm(result_paths, total=n_results):
        result = ModelExpDataResult.from_file(
            filename=result_path,
            model_class=model_class,
        )
        jsds.append(result.data_jsd)
        metric_values = [
            evaluate_model_rollout(
                rollout_comparison,
                wrapped_env,
                NodeModelWrapper(model, wrapped_env.featurize),
                batch_size,
                key,
            )
            for model in result.models
        ]
        metrics.append(jnp.median(jnp.array(metric_values)).item())

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.grid(True)

    ax.scatter(jsds, metrics, s=25, marker="x", c="r")
    ax.set_yscale("log")

    ax.set_ylabel(f"model loss for {rollout_comparison.sequence_length} steps")
    ax.set_xlabel("JSD")

    return fig, ax, (metrics, jsds)
