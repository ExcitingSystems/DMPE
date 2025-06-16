import json
import argparse
import datetime
import os
import pathlib


import numpy as np
import jax
import jax.numpy as jnp
import diffrax
from haiku import PRNGSequence

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


from dmpe.utils.signals import aprbs
from dmpe.algorithms.algorithms import excite_with_dmpe
from dmpe.models.model_utils import save_model

import dmpe.utils.env_utils.pmsm_utils as pmsm_utils
import dmpe.evaluation.scripts.pmsm.dmpe_params as dmpe_params


TARGETED_DATA_PATH = (
    pathlib.Path(__file__).parent.parent.parent.parent.parent / pathlib.Path("data") / pathlib.Path("pmsm")
)


def safe_json_dump(obj, fp):
    default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
    return json.dump(obj, fp, default=default)


def setup_env(rpm):
    env = pmsm_utils.ExcitingPMSM(
        initial_rpm=rpm,
        batch_size=1,
        saturated=True,
        LUT_motor_name="BRUSA",
        static_params={
            "p": 3,
            "r_s": 17.932e-3,
            "l_d": jnp.nan,
            "l_q": jnp.nan,
            "psi_p": 65.65e-3,
            "deadtime": 0,
            "u_dc": 400,
        },
        solver=diffrax.Tsit5(),
    )
    penalty_function = lambda observations, actions: pmsm_utils.PMSM_penalty(env, observations, actions)

    return env, penalty_function


def run_experiment(model_name, exp_idx, env, exp_params):

    seed = exp_params["seed"]
    rpm = exp_params["rpm"]
    consider_actions = exp_params["alg_params"]["consider_action_distribution"]

    print(
        "Running experiment",
        exp_idx,
        f"(seed: {int(seed)}) on the PMSM with {rpm} rpm. Considers actions? {consider_actions}",
    )

    # Check that the targeted data folder actually exist:
    results_path = TARGETED_DATA_PATH / pathlib.Path("dmpe") / pathlib.Path(model_name)
    print(f"Results will be written to: '{results_path}'.")
    assert results_path.exists(), (
        f"The expected results path '{results_path}' does not seem to exist. Please create the necessary file structure "
        + "or adapt the path."
    )

    # setup PRNG:
    key = jax.random.PRNGKey(seed=exp_params["seed"])
    data_key, model_key, loader_key, expl_key, key = jax.random.split(key, 5)
    data_rng = PRNGSequence(data_key)

    if model_name == "NODE":
        exp_params["model_params"]["key"] = model_key

    # initial guess
    proposed_actions = (
        jnp.hstack(
            [
                aprbs(exp_params["alg_params"]["n_prediction_steps"], env.batch_size, 1, 10, next(data_rng))[0]
                for _ in range(env.action_dim)
            ]
        )
        / 5
    )

    # run excitation algorithm
    observations, actions, model, density_estimate, losses, proposed_actions, _ = excite_with_dmpe(
        env, exp_params, proposed_actions, loader_key, expl_key, callback_every=None, callback=None
    )

    # save parameters
    file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(results_path / pathlib.Path(f"params_rpm_{rpm}_ca_{consider_actions}_{file_name}.json"), "w") as fp:
        safe_json_dump(exp_params, fp)

    # save observations + actions
    with open(results_path / pathlib.Path(f"data_rpm_{rpm}_ca_{consider_actions}_{file_name}.json"), "w") as fp:
        json.dump(dict(observations=observations.tolist(), actions=actions.tolist()), fp)

    model_params = exp_params["model_params"]

    if model_name == "NODE":
        model_params["key"] = model_params["key"].tolist()
    save_model(
        results_path / pathlib.Path(f"model_rpm_{rpm}_ca_{consider_actions}_{file_name}.json"),
        hyperparams=model_params,
        model=model,
    )


def main(rpm, model_name, consider_actions):

    assert 0 <= rpm <= 11000, "RPM must be between 0 and 11000."

    env, penalty_function = setup_env(rpm)

    if model_name == "NODE":
        alg_params, model_params, model_class, model_trainer_params = dmpe_params.get_NODE_params(
            consider_action_distribution=consider_actions, penalty_function=penalty_function
        )
    elif model_name == "RLS":
        alg_params, model_params, model_class, model_trainer_params = dmpe_params.get_RLS_params(
            consider_action_distribution=consider_actions, penalty_function=penalty_function
        )
    elif model_name == "PM":
        alg_params, model_params, model_class, model_trainer_params = dmpe_params.get_PM_params(
            consider_action_distribution=consider_actions, penalty_function=penalty_function
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    exp_params = dict(
        seed=None,
        rpm=float(rpm),
        n_time_steps=15_000,
        alg_params=alg_params,
        model_params=model_params,
        model_class=model_class,
        model_trainer_params=model_trainer_params,
    )
    seeds = list(np.arange(50, 61))

    for exp_idx, seed in enumerate(seeds):

        exp_params["seed"] = int(seed)
        run_experiment(model_name, exp_idx, env, exp_params)

        jax.clear_caches()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DMPE on the PMSM environment.")
    parser.add_argument("--rpm", type=float, default=2000, help="RPM of the PMSM.")
    parser.add_argument("--model", type=str, default="NODE", help="Model to use for the DMPE algorithm.")
    parser.add_argument("--consider_actions", action=argparse.BooleanOptionalAction)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id to use.")

    args = parser.parse_args()

    gpus = jax.devices()
    jax.config.update("jax_default_device", gpus[args.gpu_id])

    rpm = args.rpm
    model_name = args.model
    consider_actions = args.consider_actions is True

    main(rpm, model_name, consider_actions)
