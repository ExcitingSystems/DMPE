import json
import datetime
import argparse
import os
import pathlib

import numpy as np
import jax
import jax.numpy as jnp

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import diffrax
import optax
from haiku import PRNGSequence

import exciting_environments as excenvs

from dmpe.utils.signals import aprbs
from dmpe.utils.density_estimation import select_bandwidth, get_uniform_target_distribution
from dmpe.excitation.excitation_utils import soft_penalty
from dmpe.algorithms.algorithms import excite_with_dmpe


# file path setup
REPO_ROOT_PATH = pathlib.Path(__file__).parent.parent.parent.parent.parent
TARGETED_DATA_PATH = REPO_ROOT_PATH / pathlib.Path("data") / pathlib.Path("classical_systems")


def safe_json_dump(obj, fp):
    default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
    return json.dump(obj, fp, default=default)


parser = argparse.ArgumentParser(description="Process 'sys_name' to choose the system to experiment on.")
parser.add_argument(
    "sys_name",
    metavar="sys_name",
    type=str,
    help="The name of the environment. Options are ['pendulum', 'fluid_tank', 'cart_pole'].",
)
parser.add_argument("--gpu_id", type=int, default=0, help="GPU id to use.")

args = parser.parse_args()
sys_name = args.sys_name

gpus = jax.devices()
gpu_id = args.gpu_id
jax.config.update("jax_default_device", gpus[args.gpu_id])

### Start experiment parameters #######################################################################################
if sys_name == "pendulum":
    ## Start pendulum experiment parameters

    env_params = dict(batch_size=1, tau=2e-2, max_torque=5, g=9.81, l=1, m=1, env_solver=diffrax.Tsit5())
    env = excenvs.make(
        env_id="Pendulum-v0",
        batch_size=env_params["batch_size"],
        action_normalizations={
            "torque": excenvs.utils.MinMaxNormalization(min=-env_params["max_torque"], max=env_params["max_torque"])
        },
        static_params={"g": env_params["g"], "l": env_params["l"], "m": env_params["m"]},
        solver=env_params["env_solver"],
        tau=env_params["tau"],
    )
    alg_params = dict(
        bandwidth=None,
        n_prediction_steps=20,
        points_per_dim=21,
        grid_extend=1.05,
        excitation_optimizer=optax.adabelief(1e-1),
        n_opt_steps=10,
        start_optimizing=5,
        consider_action_distribution=True,
        penalty_function=None,
        target_distribution=None,
        clip_action=True,
        n_starts=5,
        reuse_proposed_actions=True,
    )

    alg_params["bandwidth"] = float(
        select_bandwidth(
            delta_z=2,
            dim=env.physical_state_dim + env.action_dim,
            n_g=alg_params["points_per_dim"],
            percentage=0.3,
        )
    )

    # overwrite penalty function and target distribution
    alg_params["penalty_function"] = lambda x, u: soft_penalty(a=x, a_max=1, penalty_order=2) + soft_penalty(
        a=u, a_max=1, penalty_order=2
    )
    alg_params["target_distribution"] = get_uniform_target_distribution(
        dim=3 if alg_params["consider_action_distribution"] else 2,
        points_per_dim=alg_params["points_per_dim"],
        bandwidth=alg_params["bandwidth"],
        grid_extend=alg_params["grid_extend"],
        consider_action_distribution=alg_params["consider_action_distribution"],
        penalty_function=alg_params["penalty_function"],
    )

    exp_params = dict(
        seed=None,
        n_time_steps=15_000,
        model_class=None,
        env_params=env_params,
        alg_params=alg_params,
        model_trainer_params=None,
        model_params=None,
    )
    seeds = list(np.arange(101, 131))
    ## End pendulum experiment parameters

elif sys_name == "fluid_tank":
    ## Start fluid_tank experiment parameters

    env_params = dict(
        batch_size=1,
        tau=5,
        max_height=3,
        max_inflow=0.2,
        base_area=jnp.pi,
        orifice_area=jnp.pi * 0.1**2,
        c_d=0.6,
        g=9.81,
        env_solver=diffrax.Tsit5(),
    )
    env = excenvs.make(
        "FluidTank-v0",
        physical_normalizations=dict(height=excenvs.utils.MinMaxNormalization(min=0, max=env_params["max_height"])),
        action_normalizations=dict(inflow=excenvs.utils.MinMaxNormalization(min=0, max=env_params["max_inflow"])),
        static_params=dict(
            base_area=env_params["base_area"],
            orifice_area=env_params["orifice_area"],
            c_d=env_params["c_d"],
            g=env_params["g"],
        ),
        tau=env_params["tau"],
        solver=env_params["env_solver"],
    )

    alg_params = dict(
        bandwidth=None,
        n_prediction_steps=10,
        points_per_dim=50,
        grid_extend=1.05,
        excitation_optimizer=optax.adabelief(1e-1),
        n_opt_steps=10,
        start_optimizing=5,
        consider_action_distribution=True,
        penalty_function=None,
        target_distribution=None,
        clip_action=True,
        n_starts=5,
        reuse_proposed_actions=True,
    )

    alg_params["bandwidth"] = float(
        select_bandwidth(
            delta_z=2,
            dim=env.physical_state_dim + env.action_dim,
            n_g=alg_params["points_per_dim"],
            percentage=0.3,
        )
    )

    # overwrite penalty function and target distribution
    alg_params["penalty_function"] = lambda x, u: soft_penalty(a=x, a_max=1, penalty_order=2) + soft_penalty(
        a=u, a_max=1, penalty_order=2
    )
    alg_params["target_distribution"] = get_uniform_target_distribution(
        dim=2 if alg_params["consider_action_distribution"] else 1,
        points_per_dim=alg_params["points_per_dim"],
        bandwidth=alg_params["bandwidth"],
        grid_extend=alg_params["grid_extend"],
        consider_action_distribution=alg_params["consider_action_distribution"],
        penalty_function=alg_params["penalty_function"],
    )

    exp_params = dict(
        seed=None,
        n_time_steps=15_000,
        model_class=None,
        env_params=env_params,
        alg_params=alg_params,
        model_trainer_params=None,
        model_params=None,
    )
    seeds = list(np.arange(101, 131))
    ## End fluid_tank experiment parameters

elif sys_name == "cart_pole":
    ## Start cart_pole experiment parameters

    env_params = dict(
        batch_size=1,
        tau=2e-2,
        max_force=10,
        static_params={
            "mu_p": 0.002,
            "mu_c": 0.5,
            "l": 0.5,
            "m_p": 0.1,
            "m_c": 1,
            "g": 9.81,
        },
        physical_normalizations={
            "deflection": excenvs.utils.MinMaxNormalization(min=-2.4, max=2.4),
            "velocity": excenvs.utils.MinMaxNormalization(min=-8, max=8),
            "theta": excenvs.utils.MinMaxNormalization(min=-jnp.pi, max=jnp.pi),
            "omega": excenvs.utils.MinMaxNormalization(min=-8, max=8),
        },
        env_solver=diffrax.Tsit5(),
    )
    env = excenvs.make(
        env_id="CartPole-v0",
        batch_size=env_params["batch_size"],
        action_normalizations={
            "force": excenvs.utils.MinMaxNormalization(min=-env_params["max_force"], max=env_params["max_force"])
        },
        physical_normalizations=env_params["physical_normalizations"],
        static_params=env_params["static_params"],
        solver=env_params["env_solver"],
        tau=env_params["tau"],
    )

    alg_params = dict(
        bandwidth=0.12,
        n_prediction_steps=50,
        points_per_dim=10,
        grid_extend=1.05,
        excitation_optimizer=optax.adabelief(1e-1),
        n_opt_steps=5,
        start_optimizing=5,
        consider_action_distribution=True,
        penalty_function=None,
        target_distribution=None,
        clip_action=True,
        n_starts=5,
        reuse_proposed_actions=True,
    )

    # alg_params["bandwidth"] = float(
    #     select_bandwidth(
    #         delta_z=2,
    #         dim=env.physical_state_dim + env.action_dim,
    #         n_g=alg_params["points_per_dim"],
    #         percentage=0.1,
    #     )
    # )

    # overwrite penalty function and target distribution
    alg_params["penalty_function"] = lambda x, u: soft_penalty(a=x, a_max=1, penalty_order=2) + soft_penalty(
        a=u, a_max=1, penalty_order=2
    )
    alg_params["target_distribution"] = get_uniform_target_distribution(
        dim=5 if alg_params["consider_action_distribution"] else 4,
        points_per_dim=alg_params["points_per_dim"],
        bandwidth=alg_params["bandwidth"],
        grid_extend=alg_params["grid_extend"],
        consider_action_distribution=alg_params["consider_action_distribution"],
        penalty_function=alg_params["penalty_function"],
    )

    exp_params = dict(
        seed=None,
        n_time_steps=15_000,
        model_class=None,
        env_params=env_params,
        alg_params=alg_params,
        model_trainer_params=None,
        model_params=None,
    )
    seeds = list(np.arange(101, 131))

    ## End cart_pole experiment parameters

else:
    raise NotImplementedError(f"System '{sys_name}' is unknown. Choose from ['pendulum', 'fluid_tank', 'cart_pole'].")

### End experiment parameters #########################################################################################

for exp_idx, seed in enumerate(seeds):
    print("Running experiment", exp_idx, f"(seed: {seed}) on '{sys_name}'")
    exp_params["seed"] = int(seed)

    # Check that the targeted data folder actually exist:
    results_path = TARGETED_DATA_PATH / pathlib.Path("perfect_model_dmpe") / pathlib.Path(sys_name)
    print(f"Results will be written to: '{results_path}'.")
    assert results_path.exists(), (
        f"The expected results path '{results_path}' does not seem to exist. Please create the necessary file structure "
        + "or adapt the path."
    )

    # setup PRNG
    key = jax.random.PRNGKey(seed=exp_params["seed"])
    data_key, _, _, expl_key, key = jax.random.split(key, 5)
    data_rng = PRNGSequence(data_key)

    # initial guess
    proposed_actions = jnp.hstack(
        [
            aprbs(alg_params["n_prediction_steps"], env.batch_size, 1, 10, next(data_rng))[0]
            for _ in range(env.action_dim)
        ]
    )
    # run excitation algorithm
    observations, actions, model, density_estimate, losses, proposed_actions, _ = excite_with_dmpe(
        env,
        exp_params,
        proposed_actions,
        None,
        expl_key,
    )

    # save parameters
    file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(results_path / pathlib.Path(f"params_{file_name}.json"), "w") as fp:
        safe_json_dump(exp_params, fp)

    # save observations + actions
    with open(results_path / pathlib.Path(f"data_{file_name}.json"), "w") as fp:
        json.dump(dict(observations=observations.tolist(), actions=actions.tolist()), fp)

    jax.clear_caches()

### End experiments ###################################################################################################
