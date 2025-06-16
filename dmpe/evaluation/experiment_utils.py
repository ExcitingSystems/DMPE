"""A collection of functions for loading and evaluating experiment results."""

from typing import Callable
import json
import pathlib
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from dmpe.models.model_utils import load_model
from dmpe.evaluation.plotting_utils import plot_sequence, plot_model_performance
from dmpe.evaluation.metrics_utils import default_jsd, default_ae, default_mcudsa, default_ksfc


def get_experiment_ids(results_path: pathlib.Path):
    """Get the experiment ids available in the specified directory."""
    json_file_paths = glob.glob(str(results_path / pathlib.Path("*.json")))
    identifiers = set([pathlib.Path(path).stem.split("_", maxsplit=1)[-1] for path in json_file_paths])
    return sorted(list(identifiers))


def get_organized_experiment_ids(full_results_path, force_consider_actions=False):
    """Only relevant for the PMSM.

    Get the experiment ids available in the specified directory, organized by their 'rpm' and whether
    the action distribution has been considered in the experiment.
    """
    experiment_ids = get_experiment_ids(full_results_path)
    organized_experiment_ids = {}

    for experiment_id in experiment_ids:

        ca = experiment_id.split("ca_")[-1].split("_")[0] == "True"

        if ca not in organized_experiment_ids.keys():
            organized_experiment_ids[ca] = {}

        rpm = float(experiment_id.split("rpm_")[-1].split("_")[0])
        if rpm not in organized_experiment_ids[ca].keys():
            organized_experiment_ids[ca][rpm] = []
        organized_experiment_ids[ca][rpm].append(experiment_id)

    if force_consider_actions:
        if True not in organized_experiment_ids.keys():
            print("No experiments with consider_actions=True. Regard experiments to consider actions.")
            organized_experiment_ids[True] = organized_experiment_ids[False]
            del organized_experiment_ids[False]
        else:
            del organized_experiment_ids[False]

    return organized_experiment_ids


def load_experiment_results(exp_id: str, results_path: pathlib.Path, model_class=None, to_array=True):
    """Load the experiment data for the experiment with identifier exp_id at the results path."""
    if os.path.isfile(results_path / pathlib.Path(f"params_{exp_id}.json")):
        with open(results_path / pathlib.Path(f"params_{exp_id}.json"), "rb") as fp:
            params = json.load(fp)
    else:
        params = None

    with open(results_path / pathlib.Path(f"data_{exp_id}.json"), "rb") as fp:
        data = json.load(fp)

        if to_array:
            try:
                observations = jnp.array(data["observations"])
                actions = jnp.array(data["actions"])
            except:
                try:
                    observations = jnp.array(np.concatenate(data["observations"]))
                    actions = jnp.array(np.concatenate(data["actions"]))
                except:
                    obs_dim = np.array(data["observations"][0]).shape[0]

                    observations = np.stack(data["observations"][:-obs_dim], axis=0)
                    observations = jnp.array(
                        np.concatenate([observations, np.array(data["observations"][-obs_dim:])[None, ...]])
                    )
                    actions = jnp.array(np.stack(data["actions"]))
        else:
            observations = data["observations"]
            actions = data["actions"]

    if model_class is not None:
        model = load_model(results_path / pathlib.Path(f"model_{exp_id}.json"), model_class)
        return params, observations, actions, model
    else:
        return params, observations, actions, None


def evaluate_experiment_metrics(observations, actions, metrics, featurize=None):
    """Evaluate the given observations and actions using the specified metrics."""
    results = {}

    if metrics is None:
        metrics = {
            "jsd": default_jsd,
            "ae": default_ae,
            "mcudsa": default_mcudsa,
            "ksfc": default_ksfc,
        }

    for name, metric in metrics.items():
        results[name] = metric(observations, actions).item()

    if featurize is not None:
        assert isinstance(featurize, Callable)
        for name, metric in metrics.items():
            results[f"{name}_feat"] = metric(featurize(observations), actions).item()

    return results


def evaluate_algorithm_metrics(identifiers, results_path, featurize=None):
    """Iterate over the given experiment identifiers and evaluate the corresponding observations and actions."""
    results = {}
    for identifier in identifiers:
        _, observations, actions, _ = load_experiment_results(
            exp_id=identifier, results_path=results_path, model_class=None
        )
        single_result = evaluate_experiment_metrics(observations, actions, featurize=featurize)

        if len(results.keys()) == 0:
            for key, value in single_result.items():
                results[key] = [value]
        else:
            for key, value in single_result.items():
                results[key].append(value)
    return results


def extract_metrics_over_timesteps(experiment_ids, results_path, lengths, metrics=None, slotted=False):
    """Iterate over the given experiment identifiers and evaluate the corresponding observations and actions.
    Only the first 'length' elements are considered for each 'length' in the 'lengths'-list.
    """
    all_results = []
    for idx, identifier in enumerate(experiment_ids):
        print(f"Experiment {identifier} at index {idx}")

        _, observations, actions, _ = load_experiment_results(
            exp_id=identifier,
            results_path=results_path,
            model_class=None,
        )
        if slotted:
            single_results = []

            for i in range(len(lengths) - 1):
                single_results.append(
                    evaluate_experiment_metrics(
                        observations[lengths[i] : lengths[i + 1]], actions[lengths[i] : lengths[i + 1]], metrics=metrics
                    )
                )

        else:
            single_results = [
                evaluate_experiment_metrics(observations[:N], actions[:N], metrics=metrics) for N in lengths
            ]
        metric_keys = single_results[0].keys()

        results_by_metric = {key: [] for key in metric_keys}
        for result in single_results:
            for metric_key in metric_keys:
                results_by_metric[metric_key].append(result[metric_key])

        all_results.append(results_by_metric)

    print("Reshape to results by metric...")
    all_results_by_metric = {key: [] for key in metric_keys}
    for result in all_results:
        for metric_key in metric_keys:
            all_results_by_metric[metric_key].append(result[metric_key])

    for metric_key in all_results_by_metric.keys():
        all_results_by_metric[metric_key] = jnp.stack(jnp.array(all_results_by_metric[metric_key]))

    print("Done")

    return all_results_by_metric


def extract_metrics_over_timesteps_via_interpolation(experiment_ids, results_path, target_lengths, metrics=None):
    all_raw_results = []
    all_raw_lengths = []

    all_results = []
    for idx, identifier in enumerate(experiment_ids):
        print(f"Experiment {identifier} at index {idx} via interpolation")

        _, observations, actions, _ = load_experiment_results(
            exp_id=identifier,
            results_path=results_path,
            model_class=None,
            to_array=False,
        )

        raw_lengths = [len(subsequence) for subsequence in observations]
        raw_lengths = np.cumsum(raw_lengths[:-1])

        observations = jnp.array(np.concatenate(observations))
        actions = jnp.array(np.concatenate(actions))
        raw_results = [evaluate_experiment_metrics(observations[:N], actions[:N], metrics=metrics) for N in raw_lengths]

        metric_keys = raw_results[0].keys()

        raw_results_by_metric = {key: [] for key in metric_keys}
        for result in raw_results:
            for metric_key in metric_keys:
                raw_results_by_metric[metric_key].append(result[metric_key])

        all_raw_results.append(raw_results_by_metric)
        all_raw_lengths.append(raw_lengths)

        interpolated_results_by_metric = {}
        for metric_key in metric_keys:
            interpolated_results_by_metric[metric_key] = jnp.interp(
                x=target_lengths,
                xp=raw_lengths,
                fp=jnp.array(raw_results_by_metric[metric_key]),
            )
        all_results.append(interpolated_results_by_metric)

    print("Reshape to results by metric...")
    all_results_by_metric = {key: [] for key in metric_keys}
    for result in all_results:
        for metric_key in metric_keys:
            all_results_by_metric[metric_key].append(result[metric_key])

    for metric_key in all_results_by_metric.keys():
        all_results_by_metric[metric_key] = jnp.stack(jnp.array(all_results_by_metric[metric_key]))

    print("Done")

    return {"interp": all_results_by_metric, "raw": all_raw_results, "raw_lengths": all_raw_lengths}


def quick_eval(env, identifier, results_path, model_class=None):
    """Gives a quick overview over the specified experiment."""

    params, observations, actions, model = load_experiment_results(
        exp_id=identifier, results_path=results_path, model_class=model_class
    )

    print(identifier)
    print(params["alg_params"])

    if observations.shape[0] == actions.shape[0]:
        actions = actions[:-1]

    fig, axs = plot_sequence(
        observations=observations,
        actions=actions,
        tau=env.tau,
        obs_labels=env.obs_description,
        action_labels=env.action_description,
    )
    plt.show()

    if model is not None:
        fig, axs = plot_model_performance(
            model=model,
            true_observations=observations[:1000],
            actions=actions[:999],
            tau=env.tau,
            obs_labels=env.obs_description,
            action_labels=[r"$u$"],
        )
        plt.show()
