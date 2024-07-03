import json
import pathlib
import glob

import matplotlib.pyplot as plt
import jax.numpy as jnp

from exciting_exciting_systems.models.model_utils import load_model
from exciting_exciting_systems.evaluation.plotting_utils import plot_sequence, plot_model_performance


def get_experiment_ids(results_path: pathlib.Path):
    json_file_paths = glob.glob(str(results_path / pathlib.Path("*.json")))
    identifiers = set([pathlib.Path(path).stem.split("_", maxsplit=1)[-1] for path in json_file_paths])
    return sorted(list(identifiers))


def load_experiment_results(exp_id: str, results_path: pathlib.Path, model_class=None):
    with open(results_path / pathlib.Path(f"params_{exp_id}.json"), "rb") as fp:
        params = json.load(fp)

    with open(results_path / pathlib.Path(f"data_{exp_id}.json"), "rb") as fp:
        data = json.load(fp)
        observations = jnp.array(data["observations"])
        actions = jnp.array(data["actions"])

    if model_class is not None:
        model = load_model(results_path / pathlib.Path(f"model_{exp_id}.json"), model_class)
        return params, observations, actions, model
    else:
        return params, observations, actions, None


def quick_eval_pendulum(env, identifier, results_path, model_class=None):
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
        obs_labels=[r"$\theta$", r"$\omega$"],
        action_labels=[r"$u$"],
    )
    plt.show()

    if model is not None:
        fig, axs = plot_model_performance(
            model=model,
            true_observations=observations[:1000],
            actions=actions[:999],
            tau=env.tau,
            obs_labels=[r"$\theta$", r"$\omega$"],
            action_labels=[r"$u$"],
        )
        plt.show()


def evaluate_metrics(actions, observations):
    raise NotImplementedError("TODO")
