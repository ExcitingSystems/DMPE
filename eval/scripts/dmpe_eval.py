import json
import argparse
import datetime
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import jax
import jax.numpy as jnp

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
gpus = jax.devices()
jax.config.update("jax_default_device", gpus[0])

import diffrax
from haiku import PRNGSequence

import exciting_environments as excenvs

from exciting_exciting_systems.utils.signals import aprbs
from exciting_exciting_systems.utils.density_estimation import select_bandwidth
from exciting_exciting_systems.algorithms import excite_with_dmpe
from exciting_exciting_systems.models import NeuralEulerODEPendulum, NeuralEulerODE, NeuralEulerODECartpole
from exciting_exciting_systems.models.model_utils import save_model


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

args = parser.parse_args()
sys_name = args.sys_name

### Start experiment parameters #######################################################################################
if sys_name == "pendulum":
    ## Start pendulum experiment parameters

    def featurize_theta(obs):
        """The angle itself is difficult to properly interpret in the loss as angles
        such as 1.99 * pi and 0 are essentially the same. Therefore the angle is
        transformed to sin(phi) and cos(phi) for comparison in the loss."""
        feat_obs = jnp.stack([jnp.sin(obs[..., 0] * jnp.pi), jnp.cos(obs[..., 0] * jnp.pi), obs[..., 1]], axis=-1)
        return feat_obs

    env_params = dict(batch_size=1, tau=2e-2, max_torque=5, g=9.81, l=1, m=1, env_solver=diffrax.Tsit5())
    env = excenvs.make(
        env_id="Pendulum-v0",
        batch_size=env_params["batch_size"],
        action_constraints={"torque": env_params["max_torque"]},
        static_params={"g": env_params["g"], "l": env_params["l"], "m": env_params["m"]},
        solver=env_params["env_solver"],
        tau=env_params["tau"],
    )
    alg_params = dict(
        bandwidth=None,
        n_prediction_steps=50,
        points_per_dim=50,
        action_lr=1e-1,
        n_opt_steps=10,
        rho_obs=1,
        rho_act=1,
        penalty_order=2,
        clip_action=True,
        n_starts=5,
        reuse_proposed_actions=True,
    )
    alg_params["bandwidth"] = float(
        select_bandwidth(
            delta_x=2,
            dim=env.physical_state_dim + env.action_dim,
            n_g=alg_params["points_per_dim"],
            percentage=0.3,
        )
    )

    model_trainer_params = dict(
        start_learning=alg_params["n_prediction_steps"],
        training_batch_size=128,
        n_train_steps=1,
        sequence_length=alg_params["n_prediction_steps"],
        featurize=featurize_theta,
        model_lr=1e-4,
    )
    model_params = dict(obs_dim=env.physical_state_dim, action_dim=env.action_dim, width_size=128, depth=3, key=None)

    exp_params = dict(
        seed=None,
        n_timesteps=15_000,
        model_class=NeuralEulerODEPendulum,
        env_params=env_params,
        alg_params=alg_params,
        model_trainer_params=model_trainer_params,
        model_params=model_params,
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
        physical_constraints=dict(height=env_params["max_height"]),
        action_constraints=dict(inflow=env_params["max_inflow"]),
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
        action_lr=1e-1,
        n_opt_steps=10,
        rho_obs=1,
        rho_act=1,
        penalty_order=2,
        clip_action=True,
        n_starts=5,
        reuse_proposed_actions=True,
    )
    alg_params["bandwidth"] = float(
        select_bandwidth(
            delta_x=2,
            dim=env.physical_state_dim + env.action_dim,
            n_g=alg_params["points_per_dim"],
            percentage=0.3,
        )
    )

    model_trainer_params = dict(
        start_learning=alg_params["n_prediction_steps"],
        training_batch_size=128,
        n_train_steps=1,
        sequence_length=alg_params["n_prediction_steps"],
        featurize=lambda x: x,
        model_lr=1e-4,
    )
    model_params = dict(obs_dim=env.physical_state_dim, action_dim=env.action_dim, width_size=128, depth=3, key=None)

    exp_params = dict(
        seed=None,
        n_timesteps=15_000,
        model_class=NeuralEulerODE,
        env_params=env_params,
        alg_params=alg_params,
        model_trainer_params=model_trainer_params,
        model_params=model_params,
    )
    seeds = list(np.arange(101, 131))
    ## End fluid_tank experiment parameters

elif sys_name == "cart_pole":
    ## Start cart_pole experiment parameters

    def featurize_theta_cart_pole(obs):
        """The angle itself is difficult to properly interpret in the loss as angles
        such as 1.99 * pi and 0 are essentially the same. Therefore the angle is
        transformed to sin(phi) and cos(phi) for comparison in the loss."""
        feat_obs = jnp.stack(
            [obs[..., 0], obs[..., 1], jnp.sin(obs[..., 2] * jnp.pi), jnp.cos(obs[..., 2] * jnp.pi), obs[..., 3]],
            axis=-1,
        )
        return feat_obs

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
        physical_constraints={
            "deflection": 2.4,
            "velocity": 8,
            "theta": jnp.pi,
            "omega": 8,
        },
        env_solver=diffrax.Tsit5(),
    )
    env = excenvs.make(
        env_id="CartPole-v0",
        batch_size=env_params["batch_size"],
        action_constraints={"force": env_params["max_force"]},
        physical_constraints=env_params["physical_constraints"],
        static_params=env_params["static_params"],
        solver=env_params["env_solver"],
        tau=env_params["tau"],
    )

    points_per_dim = 20

    alg_params = dict(
        bandwidth=select_bandwidth(2, 5, points_per_dim, 0.1),
        n_prediction_steps=50,
        points_per_dim=points_per_dim,
        action_lr=1e-1,
        n_opt_steps=5,
        rho_obs=1,
        rho_act=1,
        penalty_order=2,
        clip_action=True,
        n_starts=5,
        reuse_proposed_actions=True,
    )
    model_trainer_params = dict(
        start_learning=alg_params["n_prediction_steps"],
        training_batch_size=128,
        n_train_steps=10,
        sequence_length=alg_params["n_prediction_steps"],
        featurize=featurize_theta_cart_pole,
        model_lr=1e-4,
    )
    model_params = dict(obs_dim=env.physical_state_dim, action_dim=env.action_dim, width_size=128, depth=3, key=None)

    exp_params = dict(
        seed=None,
        n_timesteps=15_000,
        model_class=NeuralEulerODECartpole,
        env_params=env_params,
        alg_params=alg_params,
        model_trainer_params=model_trainer_params,
        model_params=model_params,
    )
    seeds = list(np.arange(101, 131))
    ## End cart_pole experiment parameters

### End experiment parameters #########################################################################################


### Start experiments #################################################################################################

for exp_idx, seed in enumerate(seeds):

    print("Running experiment", exp_idx, f"(seed: {seed}) on '{sys_name}'")
    exp_params["seed"] = int(seed)

    # setup PRNG
    key = jax.random.PRNGKey(seed=exp_params["seed"])
    data_key, model_key, loader_key, expl_key, key = jax.random.split(key, 5)
    data_rng = PRNGSequence(data_key)
    exp_params["model_params"]["key"] = model_key

    # initial guess
    proposed_actions = aprbs(exp_params["alg_params"]["n_prediction_steps"], env.batch_size, 1, 10, next(data_rng))[0]

    # run excitation algorithm
    observations, actions, model, density_estimate, losses, proposed_actions = excite_with_dmpe(
        env, exp_params, proposed_actions, loader_key, expl_key
    )

    # save parameters
    file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    with open(f"../results/dmpe/{sys_name}/params_{file_name}.json", "w") as fp:
        safe_json_dump(exp_params, fp)

    # save observations + actions
    with open(f"../results/dmpe/{sys_name}/data_{file_name}.json", "w") as fp:
        json.dump(dict(observations=observations.tolist(), actions=actions.tolist()), fp)

    model_params["key"] = model_params["key"].tolist()
    save_model(f"../results/dmpe/{sys_name}/model_{file_name}.json", hyperparams=model_params, model=model)

    jax.clear_caches()

### End experiments ###################################################################################################
