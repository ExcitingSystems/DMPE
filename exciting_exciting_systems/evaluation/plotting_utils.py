import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

import jax.numpy as jnp

from exciting_exciting_systems.models.model_utils import simulate_ahead


def plot_sequence(observations, actions, tau, obs_labels, action_labels, fig=None, axs=None, dotted=False):
    """Plots a given sequence of observations and actions."""

    if fig is None or axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    t = jnp.linspace(0, observations.shape[0] - 1, observations.shape[0]) * tau

    for observation_idx in range(observations.shape[-1]):
        axs[0].plot(
            t,
            jnp.squeeze(observations[..., observation_idx]),
            "." if dotted else "-",
            markersize=1,
            label=obs_labels[observation_idx],
        )

    axs[0].title.set_text("observations, timeseries")
    axs[0].legend()
    axs[0].set_ylabel(r"$\bm{x}$")
    axs[0].set_xlabel("$t$ in seconds")

    if observations.shape[-1] == 2:
        axs[1].scatter(jnp.squeeze(observations[..., 0]), jnp.squeeze(observations[..., 1]), s=1)
        axs[1].title.set_text("observation plane")
        axs[1].set_ylabel(obs_labels[1])
        axs[1].set_xlabel(obs_labels[0])
    elif observations.shape[-1] > 2:
        axs[1].scatter(jnp.squeeze(observations[..., -2]), jnp.squeeze(observations[..., -1]), s=1)
        axs[1].title.set_text("observation plane, last two obs")
        axs[1].set_ylabel(obs_labels[-1])
        axs[1].set_xlabel(obs_labels[-2])

    if actions is not None:
        if observations.shape[-1] == 1 and actions.shape[-1] == 1:
            axs[1].scatter(jnp.squeeze(actions[..., 0]), jnp.squeeze(observations[:-1, 0]), s=1)
            axs[1].title.set_text("observation $\\times$ action plane")
            axs[1].set_ylabel(obs_labels[0])
            axs[1].set_xlabel(action_labels[0])

        for action_idx in range(actions.shape[-1]):
            axs[2].plot(t[:-1], jnp.squeeze(actions[..., action_idx]), label=action_labels[action_idx])

        axs[2].title.set_text("actions, timeseries")
        axs[2].legend()
        axs[2].set_ylabel(r"$\bm{u}$")
        axs[2].set_xlabel(r"$t$ in seconds")

    for ax in axs:
        ax.grid(True)

    fig.tight_layout()
    return fig, axs


def plot_model_performance(model, true_observations, actions, tau, obs_labels, action_labels):
    """Compare the performance of the model to the ground truth data."""

    fig, axs = plot_sequence(
        observations=true_observations, actions=actions, tau=tau, obs_labels=obs_labels, action_labels=action_labels
    )

    pred_observations = simulate_ahead(model, true_observations[0, :], actions, tau)

    fig, axs = plot_sequence(
        observations=pred_observations,
        actions=None,
        tau=tau,
        obs_labels=obs_labels,
        action_labels=action_labels,
        fig=fig,
        axs=axs,
        dotted=True,
    )
    return fig, axs


def append_predictions_to_sequence_plot(
    fig, axs, starting_step, pred_observations, proposed_actions, tau, obs_labels, action_labels
):
    """Appends the future predictions to the given plot."""

    t = jnp.linspace(0, pred_observations.shape[0] - 1, pred_observations.shape[0]) * tau
    t += tau * starting_step  # start where the trajectory left off

    colors = list(mcolors.CSS4_COLORS.values())[: pred_observations.shape[-1]]
    for observation_idx, color in zip(range(pred_observations.shape[-1]), colors):
        axs[0].plot(
            t,
            jnp.squeeze(pred_observations[..., observation_idx]),
            color=color,
            label="pred " + obs_labels[observation_idx],
        )

    if pred_observations.shape[-1] == 2:
        axs[1].scatter(
            jnp.squeeze(pred_observations[..., 0]),
            jnp.squeeze(pred_observations[..., 1]),
            s=1,
            color=mcolors.CSS4_COLORS["orange"],
        )
    elif pred_observations.shape[-1] > 2:
        axs[1].scatter(
            jnp.squeeze(pred_observations[..., -2]),
            jnp.squeeze(pred_observations[..., -1]),
            s=1,
            color=mcolors.CSS4_COLORS["orange"],
        )
    elif pred_observations.shape[-1] == 1 and proposed_actions.shape[-1] == 1:
        axs[1].scatter(
            jnp.squeeze(proposed_actions[..., 0]),
            jnp.squeeze(pred_observations[:-1, 0]),
            s=1,
            color=mcolors.CSS4_COLORS["orange"],
        )

    colors = list(mcolors.CSS4_COLORS.values())[: pred_observations.shape[-1]]
    for action_idx, color in zip(range(proposed_actions.shape[-1]), colors):
        axs[2].plot(
            t[:-1],
            jnp.squeeze(proposed_actions[..., action_idx]),
            color=color,
            label="pred " + action_labels[action_idx],
        )

    return fig, axs


def plot_sequence_and_prediction(
    observations, actions, tau, obs_labels, actions_labels, model, init_obs, proposed_actions
):
    """Plots the current trajectory and appends the predictions from the optimization."""

    fig, axs = plot_sequence(
        observations=observations,
        actions=actions,
        tau=tau,
        obs_labels=obs_labels,
        action_labels=actions_labels,
    )

    pred_observations = simulate_ahead(model=model, init_obs=init_obs, actions=proposed_actions, tau=tau)

    fig, axs = append_predictions_to_sequence_plot(
        fig=fig,
        axs=axs,
        starting_step=observations.shape[0],
        pred_observations=pred_observations,
        proposed_actions=proposed_actions,
        tau=tau,
        obs_labels=obs_labels,
        action_labels=actions_labels,
    )

    return fig, axs


def plot_2d_kde_as_contourf(p_est, x, observation_labels):

    fig, axs = plt.subplots(figsize=(6.75, 6))

    grid_len_per_dim = int(np.sqrt(x.shape[0]))
    x_plot = x.reshape((grid_len_per_dim, grid_len_per_dim, 2))

    cax = axs.contourf(
        x_plot[..., 0],
        x_plot[..., 1],
        p_est.reshape(x_plot.shape[:-1]),
        antialiased=False,
        levels=50,
        alpha=0.9,
        cmap=plt.cm.coolwarm,
    )
    axs.set_xlabel(observation_labels[0])
    axs.set_ylabel(observation_labels[1])

    return fig, axs, cax


def plot_2d_kde_as_surface(p_est, x, observation_labels):

    fig = plt.figure(figsize=(6, 6))
    axs = fig.add_subplot(111, projection="3d")

    grid_len_per_dim = int(np.sqrt(x.shape[0]))
    x_plot = x.reshape((grid_len_per_dim, grid_len_per_dim, 2))

    _ = axs.plot_surface(
        x_plot[..., 0],
        x_plot[..., 1],
        p_est.reshape(x_plot.shape[:-1]),
        antialiased=False,
        alpha=0.8,
        cmap=plt.cm.coolwarm,
    )
    axs.set_xlabel(observation_labels[0])
    axs.set_ylabel(observation_labels[1])

    return fig, axs


def plot_metrics_by_sequence_length(results_by_metric, lengths):
    metric_keys = results_by_metric.keys()

    fig, axs = plt.subplots(3, figsize=(16, 12))
    colors = plt.rcParams["axes.prop_cycle"]()

    for metric_idx, metric_key in enumerate(metric_keys):
        axs[metric_idx].plot(
            lengths,
            jnp.array(results_by_metric[metric_key]),
            label=metric_key,
            color=next(colors)["color"],
        )

    [ax.grid(True) for ax in axs]
    [ax.legend() for ax in axs]

    return fig


def plot_mean_and_std_by_sequence_length(all_results_by_metric, lengths, use_log=False):
    metric_keys = all_results_by_metric.keys()

    fig, axs = plt.subplots(3, figsize=(16, 12), sharex=True)
    colors = plt.rcParams["axes.prop_cycle"]()

    for metric_idx, metric_key in enumerate(metric_keys):
        mean = jnp.mean(all_results_by_metric[metric_key], axis=0)
        std = jnp.std(all_results_by_metric[metric_key], axis=0)

        c = next(colors)["color"]

        axs[metric_idx].plot(
            lengths,
            jnp.log(mean) if use_log else mean,
            label=("log " if use_log else "") + metric_key,
            color=c,
        )
        axs[metric_idx].fill_between(
            lengths,
            jnp.log(mean - std) if use_log else mean - std,
            jnp.log(mean + std) if use_log else mean + std,
            color=c,
            alpha=0.2,
        )
        # axs[metric_idx].set_ylabel(("log " if use_log else "") + metric_key)
    axs[-1].set_xlabel("timesteps")
    axs[-1].set_xlim(lengths[0] - 100, lengths[-1] + 100)
    [ax.grid(True) for ax in axs]
    [ax.legend() for ax in axs]
    plt.tight_layout()

    return fig


def plot_metrics_by_sequence_length_for_all_algos(data_per_algo, lengths, algo_names, use_log=False):
    assert len(data_per_algo) == len(algo_names), "Mismatch in number of algo results and number of algo names"

    metric_keys = data_per_algo[0].keys()

    fig, axs = plt.subplots(3, figsize=(16, 12), sharex=True)
    colors = plt.rcParams["axes.prop_cycle"]()

    for algo_name, data in zip(algo_names, data_per_algo):
        c = next(colors)["color"]

        for metric_idx, metric_key in enumerate(metric_keys):
            mean = jnp.mean(data[metric_key], axis=0)
            std = jnp.std(data[metric_key], axis=0)

            axs[metric_idx].plot(
                lengths,
                jnp.log(mean) if use_log else mean,
                label=algo_name,
                color=c,
            )
            axs[metric_idx].fill_between(
                lengths,
                jnp.log(mean - std) if use_log else mean - std,
                jnp.log(mean + std) if use_log else mean + std,
                color=c,
                alpha=0.2,
            )
            # axs[metric_idx].set_ylabel(("log " if use_log else "") + metric_key)

    for idx, metric_key in enumerate(metric_keys):
        axs[idx].title.set_text(("log " if use_log else "") + metric_key)

    axs[-1].set_xlabel("timesteps")
    axs[-1].set_xlim(lengths[0] - 100, lengths[-1] + 100)
    [ax.grid(True) for ax in axs]
    [ax.legend() for ax in axs]
    plt.tight_layout()

    return fig
