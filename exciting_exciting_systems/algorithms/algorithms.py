from typing import Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from exciting_exciting_systems.algorithms.algorithm_utils import interact_and_observe
from exciting_exciting_systems.evaluation.plotting_utils import plot_sequence_and_prediction
from exciting_exciting_systems.excitation import loss_function, Exciter
from exciting_exciting_systems.models.model_training import ModelTrainer
from exciting_exciting_systems.utils.density_estimation import DensityEstimate, build_grid
from exciting_exciting_systems.utils.metrics import JSDLoss


def excite_and_fit(
    n_timesteps: int,
    env,
    model: eqx.Module,
    obs: jnp.ndarray,
    state,
    proposed_actions: jnp.ndarray,
    exciter: Exciter,
    model_trainer: ModelTrainer,
    density_estimate: DensityEstimate,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    opt_state_model: optax.OptState,
    loader_key: jax.random.PRNGKey,
    expl_key: jax.random.PRNGKey,
    plot_every: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, eqx.Module, DensityEstimate]:
    """
    Main algorithm to throw at a given (unknown) system and generate informative data from that system.

    Args:
        n_timesteps (int): The number of timesteps to run the algorithm for.
        env: The environment object representing the system.
        model (eqx.Module): The model used for prediction.
        obs (jnp.ndarray): The initial observation of the system.
        state: The initial state of the system.
        proposed_actions (jnp.ndarray): The proposed actions for exploration.
        exciter (Exciter): The exciter object responsible for choosing actions.
        model_trainer (ModelTrainer): The model trainer object responsible for training the model.
        density_estimate (DensityEstimate): The density estimate used for exploration.
        observations (jnp.ndarray): The history of observations.
        actions (jnp.ndarray): The history of actions.
        opt_state_model (optax.OptState): The optimizer state for the model.
        loader_key (jax.random.PRNGKey): The key used for loading data.
        plot_every (int): The frequency at which to plot the sequence and prediction.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, eqx.Module, DensityEstimate]: A tuple containing the updated history of observations,
        the updated history of actions, the updated model, and the updated density estimate.
    """
    prediction_losses = []
    data_losses = []

    for k in tqdm(range(n_timesteps)):
        action, proposed_actions, density_estimate, prediction_loss = exciter.choose_action(
            obs=obs,
            model=model,
            density_estimate=density_estimate,
            proposed_actions=proposed_actions,
            expl_key=expl_key,
        )
        prediction_losses.append(prediction_loss)

        obs, state, actions, observations = interact_and_observe(
            env=env, k=jnp.array([k]), action=action, state=state, actions=actions, observations=observations
        )

        if model_trainer is not None:
            if k > model_trainer.start_learning:
                model, opt_state_model, loader_key = model_trainer.fit(
                    model=model,
                    k=jnp.array([k]),
                    observations=observations,
                    actions=actions,
                    opt_state=opt_state_model,
                    loader_key=loader_key,
                )

        data_loss = JSDLoss(
            density_estimate.p / jnp.sum(density_estimate.p),
            exciter.target_distribution / jnp.sum(exciter.target_distribution),
        )
        data_losses.append(data_loss)

        if k % plot_every == 0 and k > 0:
            print("last input opt loss:", prediction_losses[-1])
            print("current data loss:", data_loss)
            fig, axs = plot_sequence_and_prediction(
                observations=observations[: k + 2, :],
                actions=actions[: k + 1, :],
                tau=exciter.tau,
                obs_labels=env.obs_description,
                actions_labels=[r"$u$"],
                model=model,
                init_obs=obs,
                proposed_actions=proposed_actions,
            )
            plt.show()

            plt.plot(np.log(data_losses))
            plt.grid(True)
            plt.show()

    return observations, actions, model, density_estimate, prediction_losses, proposed_actions


def excite_with_dmpe(
    env,
    exp_params: dict,
    proposed_actions: jnp.ndarray,
    loader_key: jax.random.PRNGKey,
    expl_key: jax.random.PRNGKey,
    plot_every=None,
):
    """
    Excite the system using the Differentiable Model Predictive Excitation (DMPE) algorithm.

    Args:
        env: The environment object representing the system.
        exp_params: The experiment parameters.
        proposed_actions: The proposed actions for exploration.
        model_key: The key for initializing the model.
        loader_key: The key used for loading data.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, eqx.Module, DensityEstimate]: A tuple containing the history of observations,
        the history of actions, the trained model, and the density estimate.
    """
    dim_obs_space = env.physical_state_dim  # assumes fully observable system
    dim_action_space = env.action_dim
    dim = dim_obs_space + dim_action_space
    n_grid_points = exp_params["alg_params"]["points_per_dim"] ** dim

    # setup x_0 / y_0
    obs, state = env.reset()
    obs = obs[0]

    # setup memory variables
    observations = jnp.zeros((exp_params["n_timesteps"], dim_obs_space))
    observations = observations.at[0].set(obs)
    actions = jnp.zeros((exp_params["n_timesteps"] - 1, dim_action_space))

    exciter = Exciter(
        loss_function=loss_function,
        grad_loss_function=jax.value_and_grad(loss_function, argnums=(2)),
        excitation_optimizer=optax.adabelief(exp_params["alg_params"]["action_lr"]),
        tau=env.tau,
        n_opt_steps=exp_params["alg_params"]["n_opt_steps"],
        target_distribution=jnp.ones(shape=(n_grid_points, 1)) * 1 / (1 - (-1)) ** dim,
        rho_obs=exp_params["alg_params"]["rho_obs"],
        rho_act=exp_params["alg_params"]["rho_act"],
        penalty_order=exp_params["alg_params"]["penalty_order"],
        clip_action=exp_params["alg_params"]["clip_action"],
    )

    if exp_params["model_trainer_params"] is None or exp_params["model_params"] is None:
        model_trainer = None
        model = exp_params["model_env_wrapper"](env)
        opt_state_model = None
    else:
        model_trainer = ModelTrainer(
            start_learning=exp_params["model_trainer_params"]["start_learning"],
            training_batch_size=exp_params["model_trainer_params"]["training_batch_size"],
            n_train_steps=exp_params["model_trainer_params"]["n_train_steps"],
            sequence_length=exp_params["model_trainer_params"]["sequence_length"],
            featurize=exp_params["model_trainer_params"]["featurize"],
            model_optimizer=optax.adabelief(exp_params["model_trainer_params"]["model_lr"]),
            tau=env.tau,
        )
        model = exp_params["model_class"](**exp_params["model_params"])
        opt_state_model = model_trainer.model_optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    density_estimate = DensityEstimate(
        p=jnp.zeros([n_grid_points, 1]),
        x_g=build_grid(dim, low=-1, high=1, points_per_dim=exp_params["alg_params"]["points_per_dim"]),
        bandwidth=jnp.array([exp_params["alg_params"]["bandwidth"]]),
        n_observations=jnp.array([0]),
    )

    observations, actions, model, density_estimate, losses, proposed_actions = excite_and_fit(
        n_timesteps=exp_params["n_timesteps"],
        env=env,
        model=model,
        obs=obs,
        state=state,
        proposed_actions=proposed_actions,
        exciter=exciter,
        model_trainer=model_trainer,
        density_estimate=density_estimate,
        observations=observations,
        actions=actions,
        opt_state_model=opt_state_model,
        loader_key=loader_key,
        expl_key=expl_key,
        plot_every=plot_every if plot_every is not None else exp_params["n_timesteps"] + 1,
    )

    return observations, actions, model, density_estimate, losses, proposed_actions
