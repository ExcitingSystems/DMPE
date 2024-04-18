import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp


def plot_sequence(observations, actions, tau, obs_labels, action_labels, ):
    
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
    return fig
