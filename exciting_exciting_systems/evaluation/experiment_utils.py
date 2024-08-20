from typing import Callable
import json
import pathlib
import glob

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from exciting_exciting_systems.models.model_utils import load_model
from exciting_exciting_systems.evaluation.plotting_utils import plot_sequence, plot_model_performance
from exciting_exciting_systems.evaluation.metrics_utils import default_jsd, default_ae, default_mcudsa, default_ksfc


def get_experiment_ids(results_path: pathlib.Path):
    json_file_paths = glob.glob(str(results_path / pathlib.Path("*.json")))
    identifiers = set([pathlib.Path(path).stem.split("_", maxsplit=1)[-1] for path in json_file_paths])
    return sorted(list(identifiers))


def load_experiment_results(exp_id: str, results_path: pathlib.Path, model_class=None, to_array=True):
    with open(results_path / pathlib.Path(f"params_{exp_id}.json"), "rb") as fp:
        params = json.load(fp)

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


def evaluate_metrics(algorithm_names, n_results, results_parent_path, featurize):
    """Gathers the last 'n_results' experiments for differnet algorithms and evaluates the metrics."""

    results = {}
    for algorithm_name in algorithm_names:
        results_path = results_parent_path / pathlib.Path(algorithm_name)
        algorithm_results = evaluate_algorithm_metrics(
            identifiers=get_experiment_ids(results_path)[-n_results:],
            results_path=results_path,
            featurize=featurize,
        )
        results[algorithm_name] = algorithm_results
    return results


def extract_metrics_over_timesteps(experiment_ids, results_path, lengths, metrics=None, slotted=False):
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


def quick_eval(env, identifier, results_path, model_class=None):
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
        action_labels=[r"$u$"],
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
