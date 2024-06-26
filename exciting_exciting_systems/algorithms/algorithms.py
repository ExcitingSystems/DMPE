import matplotlib.pyplot as plt
from tqdm import tqdm

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from exciting_exciting_systems.algorithms.algorithm_utils import interact_and_observe
from exciting_exciting_systems.models.model_training import precompute_starting_points, fit
from exciting_exciting_systems.evaluation.plotting_utils import plot_sequence_and_prediction
from exciting_exciting_systems.excitation import loss_function, Exciter
from exciting_exciting_systems.models.model_training import ModelTrainer
from exciting_exciting_systems.models import NeuralEulerODEPendulum
from exciting_exciting_systems.utils.density_estimation import DensityEstimate, build_grid_3d


def excite_and_fit(
    n_timesteps,
    env,
    model,
    obs,
    state,
    proposed_actions,
    exciter,
    model_trainer,
    density_estimate,
    observations,
    actions,
    opt_state_model,
    loader_key,
    plot_every,
):
    """Main algorithm to throw at a given (unknown) system and generate informative data from that system.

    Args:

    Returns:

    """
    for k in tqdm(range(n_timesteps)):
        action, proposed_actions, density_estimate = exciter.choose_action(
            obs=obs, model=model, density_estimate=density_estimate, proposed_actions=proposed_actions
        )

        obs, state, actions, observations = interact_and_observe(
            env=env, k=jnp.array([k]), action=action, obs=obs, state=state, actions=actions, observations=observations
        )

        if k > model_trainer.start_learning:
            model, opt_state_model, loader_key = model_trainer.fit(
                model=model,
                k=jnp.array([k]),
                observations=observations,
                actions=actions,
                opt_state=opt_state_model,
                loader_key=loader_key,
            )

        if k % plot_every == 0 and k > 0:
            fig, axs = plot_sequence_and_prediction(
                observations=observations[: k + 2, :],
                actions=actions[: k + 1, :],
                tau=exciter.tau,
                obs_labels=[r"$\theta$", r"$\omega$"],
                actions_labels=[r"$u$"],
                model=model,
                init_obs=obs,
                proposed_actions=proposed_actions,
            )
            plt.show()

    return observations, actions, model, density_estimate


def excite_with_dmpe(
    env,
    exp_params,
    proposed_actions,
    model_key,
    loader_key,
):
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
        grad_loss_function=jax.grad(loss_function, argnums=(2)),
        excitation_optimizer=optax.adabelief(exp_params["alg_params"]["action_lr"]),
        tau=env.tau,
        n_opt_steps=exp_params["alg_params"]["n_opt_steps"],
        target_distribution=jnp.ones(shape=(n_grid_points, 1)) * 1 / (1 - (-1)) ** dim,
        rho_obs=exp_params["alg_params"]["rho_obs"],
        rho_act=exp_params["alg_params"]["rho_act"],
    )

    model_trainer = ModelTrainer(
        start_learning=exp_params["model_trainer_params"]["start_learning"],
        training_batch_size=exp_params["model_trainer_params"]["training_batch_size"],
        n_train_steps=exp_params["model_trainer_params"]["n_train_steps"],
        sequence_length=exp_params["model_trainer_params"]["sequence_length"],
        featurize=exp_params["model_trainer_params"]["featurize"],
        model_optimizer=optax.adabelief(exp_params["model_trainer_params"]["model_lr"]),
        tau=env.tau,
    )

    density_estimate = DensityEstimate(
        p=jnp.zeros([n_grid_points, 1]),
        x_g=build_grid_3d(low=-1, high=1, points_per_dim=exp_params["alg_params"]["points_per_dim"]),
        bandwidth=jnp.array([exp_params["alg_params"]["bandwidth"]]),
        n_observations=jnp.array([0]),
    )

    model = NeuralEulerODEPendulum(**exp_params["model_params"])
    opt_state_model = model_trainer.model_optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    observations, actions, model, density_estimate = excite_and_fit(
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
        plot_every=exp_params["n_timesteps"] + 1,
    )

    return observations, actions, model, density_estimate
