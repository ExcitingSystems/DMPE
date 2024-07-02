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


def load_experiment_results(exp_id: str, model_class, results_path: pathlib.Path):
    with open(results_path / pathlib.Path(f"params_{exp_id}.json"), "rb") as fp:
        params = json.load(fp)

    with open(results_path / pathlib.Path(f"data_{exp_id}.json"), "rb") as fp:
        data = json.load(fp)
        observations = jnp.array(data["observations"])
        actions = jnp.array(data["actions"])

    model = load_model(results_path / pathlib.Path(f"model_{exp_id}.json"), model_class)

    return params, observations, actions, model


def quick_eval_pendulum(env, identifier, model_class, results_path):
    params, observations, actions, model = load_experiment_results(
        exp_id=identifier, model_class=model_class, results_path=results_path
    )

    print(identifier)
    print(params["alg_params"])

    fig, axs = plot_sequence(
        observations=observations,
        actions=actions,
        tau=env.tau,
        obs_labels=[r"$\theta$", r"$\omega$"],
        action_labels=[r"$u$"],
    )
    plt.show()

    fig, axs = plot_model_performance(
        model=model,
        true_observations=observations[:1000],
        actions=actions[:999],
        tau=env.tau,
        obs_labels=[r"$\theta$", r"$\omega$"],
        action_labels=[r"$u$"],
    )
    plt.show()
