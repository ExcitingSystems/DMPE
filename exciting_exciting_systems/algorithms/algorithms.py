import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import jax
import jax.numpy as jnp
import equinox as eqx

from exciting_exciting_systems.evaluation.plotting_utils import plot_sequence_and_prediction
from exciting_exciting_systems.optimization import excite
from exciting_exciting_systems.models.model_training import fit


def excite_and_fit(
        n_timesteps,
        env,
        grad_loss_function,
        obs,
        state,
        proposed_actions,
        density_estimate,
        tau,
        target_distribution,
        n_prediction_steps,
        training_batch_size,
        model,
        n_train_steps,
        sequence_length,
        observations,
        actions,
        featurize,
        solver_prediction,
        solver_model,
        opt_state_model,
        loader_key
):
    """Main algorithm to throw at a given (unknown) system and generate informative data from that system.

    Args:

    Returns:

    """
    for k in tqdm(range(n_timesteps)):
        obs, state, actions, observations, proposed_actions, density_estimate = excite(
            env,
            actions,
            observations,
            grad_loss_function,
            proposed_actions,
            model,
            solver_prediction,
            obs,
            state,
            density_estimate,
            tau,
            target_distribution
        )

        if k > n_prediction_steps:
            starting_points, loader_key = precompute_starting_points(
                n_train_steps, jnp.array([k]), sequence_length, training_batch_size, loader_key
            )

            model, opt_state_model = fit(
                model,
                n_train_steps,
                starting_points,
                sequence_length,
                observations, 
                actions,
                tau,
                featurize,
                solver_model,
                opt_state_model
            )

        if k % 500 == 0 and k > 0:
            fig, axs = plot_sequence_and_prediction(
                observations=observations[:k+2,:],
                actions=actions[:k+1,:],
                tau=tau,
                obs_labels=[r"$\theta$", r"$\omega$"],
                actions_labels=[r"$u$"],
                model=model,
                init_obs=obs[0, :],
                init_state=state[0, :],
                proposed_actions=proposed_actions[0, :]
            )
            plt.show()

    return observations, actions, model
