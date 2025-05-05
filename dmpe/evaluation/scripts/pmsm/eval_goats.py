import json
import argparse
import datetime
import os


import numpy as np
import jax
import jax.numpy as jnp
import diffrax
from haiku import PRNGSequence

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

jax.config.update("jax_platform_name", "cpu")


from dmpe.utils.signals import aprbs
from dmpe.related_work.algorithms import excite_with_iGOATS
from dmpe.models.model_utils import save_model

import dmpe.utils.env_utils.pmsm_utils as pmsm_utils
import dmpe_params

from eval_dmpe import setup_env, safe_json_dump
from goats_params import get_alg_params


def run_experiment(exp_idx, env, exp_params):

    seed = exp_params["seed"]
    rpm = exp_params["rpm"]
    consider_actions = exp_params["alg_params"]["consider_action_distribution"]

    print(
        "Running experiment",
        exp_idx,
        f"(seed: {int(seed)}) on the PMSM with {rpm} rpm. Considers actions? {consider_actions}",
    )

    # run excitation algorithm
    if not consider_actions:
        raise NotImplementedError("This code does not has no option to not take actions into account")

    alg_params = exp_params["alg_params"]
    observations, actions = excite_with_iGOATS(
        n_time_steps=exp_params["n_time_steps"],
        env=env,
        prediction_horizon=alg_params["prediction_horizon"],
        application_horizon=alg_params["application_horizon"],
        bounds_amplitude=alg_params["bounds_amplitude"],
        bounds_duration=alg_params["bounds_duration"],
        population_size=alg_params["population_size"],
        n_generations=alg_params["n_generations"],
        featurize=alg_params["featurize"],
        rng=np.random.default_rng(seed),
        compress_data=alg_params["compress_data"],
        compression_target_N=alg_params["compression_target_N"],
        compression_feat_dim=alg_params["compression_feat_dim"],
        compression_dist_th=alg_params["compression_dist_th"],
        penalty_function=alg_params["penalty_function"],
        plot_subsequences=False,
    )

    observations = [obs.tolist() for obs in observations]
    actions = [act.tolist() for act in actions]

    # save parameters
    file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    with open(f"./results/igoats/params_rpm_{rpm}_ca_{consider_actions}_{file_name}.json", "w") as fp:
        safe_json_dump(exp_params, fp)

    # save observations + actions
    with open(f"./results/igoats/data_rpm_{rpm}_ca_{consider_actions}_{file_name}.json", "w") as fp:
        json.dump(dict(observations=observations, actions=actions), fp)


def main(rpm, consider_actions):

    assert 0 <= rpm <= 11000, "RPM must be between 0 and 11000."

    env, penalty_function = setup_env(rpm)

    alg_params = get_alg_params(consider_actions, penalty_function)

    exp_params = dict(
        seed=None,
        rpm=float(rpm),
        n_time_steps=15_000,
        alg_params=alg_params,
        model_params=None,
        model_class=None,
        model_trainer_params=None,
        model_env_wrapper=None,
    )
    seeds = list(np.arange(50, 61))

    for exp_idx, seed in enumerate(seeds):

        exp_params["seed"] = int(seed)
        run_experiment(exp_idx, env, exp_params)

        jax.clear_caches()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DMPE on the PMSM environment.")
    parser.add_argument("--rpm", type=float, default=2000, help="RPM of the PMSM.")
    parser.add_argument("--consider_actions", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    jax.config.update("jax_platform_name", "cpu")

    rpm = args.rpm
    consider_actions = args.consider_actions is True

    main(rpm, consider_actions)
