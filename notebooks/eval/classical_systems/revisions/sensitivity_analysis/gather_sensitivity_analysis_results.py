from functools import partial
import argparse
import pathlib

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from tqdm import tqdm
import pandas as pd
import jax

from dmpe.data_management import DataPaths
from dmpe.data_management import ClassicalSystems as Systems
from dmpe.utils.density_estimation import select_bandwidth
from dmpe.evaluation.experiment_utils import get_experiment_ids, load_experiment_results, evaluate_experiment_metrics
from dmpe.evaluation.experiment_utils import default_jsd, default_ae, default_mcudsa, default_ksfc, default_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "algo_name",
        metavar="algo_name",
        type=str,
        help="The name of the algorithm. Options are ['dmpe', 'perfect_model_dmpe']",
    )

    parser.add_argument(
        "system_name",
        metavar="system_name",
        type=str,
        help="The name of the environment. Options are ['pendulum', 'fluid_tank', 'cart_pole'].",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id to use.")

    args = parser.parse_args()

    return args


def get_parameterized_metrics(system_name: Systems):
    if system_name == Systems.FLUID_TANK:
        metrics = {
            "jsd": partial(default_jsd, points_per_dim=50, bandwidth=select_bandwidth(2, 2, 50, 0.3).item()),
            "ae": default_ae,
            "mcudsa": partial(default_mcudsa, points_per_dim=50),
            "ksfc": partial(default_ksfc, points_per_dim=50, eps=1e-6),
            "df": partial(default_df, points_per_dim=50),
        }
    elif system_name == Systems.PENDULUM:
        metrics = {
            "jsd": partial(default_jsd, points_per_dim=50, bandwidth=select_bandwidth(2, 3, 50, 0.3).item()),
            "ae": default_ae,
            "mcudsa": partial(default_mcudsa, points_per_dim=50),
            "ksfc": partial(default_ksfc, points_per_dim=50, eps=1e-6),
            "df": partial(default_df, points_per_dim=15),
        }
    elif system_name == Systems.CART_POLE:
        metrics = {
            "jsd": partial(default_jsd, points_per_dim=20, bandwidth=select_bandwidth(2, 5, 20, 0.1).item()),
            "ae": default_ae,
            "mcudsa": partial(default_mcudsa, points_per_dim=20),
            "ksfc": partial(default_ksfc, points_per_dim=20, variance=0.1, eps=1e-6),
            "df": partial(default_df, points_per_dim=7),
        }

    return metrics


if __name__ == "__main__":
    args = parse_args()

    algo_name = args.algo_name

    gpus = jax.devices()
    gpu_id = args.gpu_id
    jax.config.update("jax_default_device", gpus[args.gpu_id])

    if args.system_name == "all":
        system_names = [system_name for system_name in Systems]
    else:
        system_names = [Systems[args.system_name.upper()]]

    for system_name in system_names:
        print(f"Gathering results for system '{system_name}' and algorithm '{algo_name}'.")

        # get metrics specific for the given system
        metrics = get_parameterized_metrics(system_name)

        print("used metrics: ", metrics)

        # get all currently stored results for the sensitivity analysis
        result_path = (
            DataPaths().sensitivity_analysis_experiments
            / pathlib.Path(algo_name)
            / pathlib.Path(system_name.name.lower())
        )
        exp_ids = get_experiment_ids(result_path)

        # predefined structure for the results data frame
        data_dict = {
            "exp_id": [],
            "a": [],
            "bandwidth": [],
            "seed": [],
            "jsd": [],
            "ae": [],
            "mcudsa": [],
            "ksfc": [],
            "df": [],
        }

        # iterate over results
        for exp_id in tqdm(exp_ids):
            params, observations, actions, _ = load_experiment_results(exp_id, result_path)
            data_dict["exp_id"].append(exp_id)
            data_dict["bandwidth"].append(params["alg_params"]["bandwidth"])
            data_dict["seed"].append(params["seed"])
            data_dict["a"].append(None)

            metric_values = evaluate_experiment_metrics(
                observations,
                actions,
                metrics=metrics,
            )
            for metric_key, metric_value in metric_values.items():
                data_dict[metric_key].append(metric_value)

        results_df = pd.DataFrame(data_dict)

        # store the df to disk:
        result_path = DataPaths().sensitivity_analysis_experiments / pathlib.Path(
            f"sensitivity_analysis_data_{system_name.name.lower()}_{algo_name}.pkl"
        )

        results_df.to_pickle(result_path)
        print(f"Stored resulting data frame for system '{system_name}' and algorithm '{algo_name}' to disk.")
