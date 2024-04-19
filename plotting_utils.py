import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

import jax
import jax.numpy as jnp

from model_utils import simulate_ahead


def plot_sequence(observations, actions, tau, obs_labels, action_labels):
    """Plots a given sequence of observations and actions."""

    fig, axs = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(18, 6),
        sharey=True
    )

    t = jnp.linspace(0, observations.shape[1]-1, observations.shape[1]) * tau

    for observation_idx in range(observations.shape[-1]):
        axs[0].plot(t, jnp.squeeze(observations[..., observation_idx]), label=obs_labels[observation_idx])

    axs[0].title.set_text("observations, timeseries")
    axs[0].legend()
    axs[0].set_xlabel(r"time $t$ in seconds")

    if observations.shape[-1] == 2:
        axs[1].scatter(jnp.squeeze(observations[..., 0]), jnp.squeeze(observations[..., 1]), s=1)
        axs[1].title.set_text("observations, together")

    for action_idx in range(actions.shape[-1]):
        axs[2].plot(t, jnp.squeeze(actions[..., action_idx]), label=action_labels[action_idx])
    
    axs[2].title.set_text("actions, timeseries")
    axs[2].legend()
    axs[2].set_xlabel(r"time $t$ in seconds")

    for ax in axs:
        ax.grid()

    fig.tight_layout()
    return fig, axs


def append_predictions_to_sequence_plot(
        fig,
        axs,
        starting_step,
        pred_observations,
        proposed_actions,
        tau,
        obs_labels,
        action_labels
    ):
    """Appends the future predictions to the given plot."""

    t = jnp.linspace(0, pred_observations.shape[1]-1, pred_observations.shape[1]) * tau
    t += tau * starting_step  # start where the trajectory left off


    colors = list(mcolors.CSS4_COLORS.values())[:pred_observations.shape[-1]]
    for observation_idx, color in zip(range(pred_observations.shape[-1]), colors):
        axs[0].plot(t, jnp.squeeze(pred_observations[..., observation_idx]), color=color, label="pred " + obs_labels[observation_idx])

    if pred_observations.shape[-1] == 2:
        axs[1].scatter(jnp.squeeze(pred_observations[..., 0]), jnp.squeeze(pred_observations[..., 1]), s=1, color=mcolors.CSS4_COLORS["orange"])

    colors = list(mcolors.CSS4_COLORS.values())[:pred_observations.shape[-1]]
    for action_idx, color in zip(range(proposed_actions.shape[-1]), colors):
        axs[2].plot(t, jnp.squeeze(proposed_actions[..., action_idx]), color=color, label="pred " + action_labels[action_idx])

    return fig, axs


def plot_sequence_and_prediction(
        observations,
        actions,
        tau,
        obs_labels,
        actions_labels,
        model,
        n_prediction_steps,
        init_obs,
        init_state,
        proposed_actions
    ):
    """Plots the current trajectory and appends the predictions from the optimization."""
    
    fig, axs = plot_sequence(
        observations=observations,
        actions=actions,
        tau=tau,
        obs_labels=obs_labels,
        action_labels=actions_labels,
    )

    pred_observations = simulate_ahead(
        model=model,
        n_steps=n_prediction_steps,
        obs=init_obs,
        state=init_state,
        actions=proposed_actions
    )

    fig, axs = append_predictions_to_sequence_plot(
        fig=fig,
        axs=axs,
        starting_step=observations.shape[1],
        pred_observations=pred_observations,
        proposed_actions=proposed_actions,
        tau=tau,
        obs_labels=obs_labels,
        action_labels=actions_labels,
    )

    return fig, axs
