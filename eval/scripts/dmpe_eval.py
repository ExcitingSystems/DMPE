import json
import datetime
import os

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
from exciting_exciting_systems.algorithms import excite_with_dmpe
from exciting_exciting_systems.models import NeuralEulerODEPendulum
from exciting_exciting_systems.models.model_utils import save_model


def featurize_theta(obs):
    """The angle itself is difficult to properly interpret in the loss as angles
    such as 1.99 * pi and 0 are essentially the same. Therefore the angle is
    transformed to sin(phi) and cos(phi) for comparison in the loss."""
    feat_obs = jnp.stack([jnp.sin(obs[..., 0] * jnp.pi), jnp.cos(obs[..., 0] * jnp.pi), obs[..., 1]], axis=-1)
    return feat_obs


def safe_json_dump(obj, fp):
    default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
    return json.dump(obj, fp, default=default)


### Start Experiment parameters
env_params = dict(batch_size=1, tau=2e-2, max_torque=5, g=9.81, l=1, m=1, env_solver=diffrax.Euler())
env = excenvs.make(
    env_id="Pendulum-v0",
    batch_size=env_params["batch_size"],
    action_constraints={"torque": env_params["max_torque"]},
    static_params={"g": env_params["g"], "l": env_params["l"], "m": env_params["m"]},
    solver=env_params["env_solver"],
    tau=env_params["tau"],
)


alg_params = dict(
    bandwidth=0.05,
    n_prediction_steps=50,
    points_per_dim=50,
    action_lr=1e-1,
    n_opt_steps=25,
    rho_obs=1,
    rho_act=1,
    penalty_order=2,
    clip_action=False,
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

seeds = list(np.arange(1, 101))
### End Experiment parameters

for exp_idx, seed in enumerate(seeds):

    print("Experiment:", exp_idx, f"(seed: {seed})")

    exp_params = dict(
        seed=seed,
        n_timesteps=15_000,
        model_class=NeuralEulerODEPendulum,
        env_params=env_params,
        alg_params=alg_params,
        model_trainer_params=model_trainer_params,
    )

    # setup PRNG
    key = jax.random.PRNGKey(seed=exp_params["seed"])
    data_key, model_key, loader_key, expl_key, key = jax.random.split(key, 5)
    data_rng = PRNGSequence(data_key)

    model_params["key"] = model_key
    exp_params["model_params"] = model_params

    # initial guess
    proposed_actions = aprbs(alg_params["n_prediction_steps"], env.batch_size, 1, 10, next(data_rng))[0]

    # run excitation algorithm
    observations, actions, model, density_estimate, losses, proposed_actions = excite_with_dmpe(
        env, exp_params, proposed_actions, loader_key, expl_key
    )

    # save parameters
    file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    with open(f"../results/dmpe/params_{file_name}.json", "w") as fp:
        safe_json_dump(exp_params, fp)

    # save observations + actions
    with open(f"../results/dmpe/data_{file_name}.json", "w") as fp:
        json.dump(dict(observations=observations.tolist(), actions=actions.tolist()), fp)

    model_params["key"] = model_params["key"].tolist()
    save_model(f"../results/dmpe/model_{file_name}.json", hyperparams=model_params, model=model)

    jax.clear_caches()
