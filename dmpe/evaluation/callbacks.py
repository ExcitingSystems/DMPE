import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from dmpe.models.model_utils import simulate_ahead, simulate_ahead_with_env
from dmpe.models.model_training import precompute_starting_points, load_single_batch
from dmpe.evaluation.plotting_utils import plot_sequence_and_prediction, plot_model_performance


def callback_template(**kwargs):
    """Template callback function to show what keyword arguments are available.

    Args:
        **kwargs: Keyword arguments passed to the callback.
    """
    for kwargs_key, kwargs_value in kwargs.items():
        print(f"{kwargs_key}: {type(kwargs_value)}")


def plot_sequence_callback(**kwargs):
    """Callback that plots the sequence of observations and actions and the current
    prediction of the model that will be used to initialize the optimization of the
    next action.
    """
    k = kwargs["k"].item()
    fig, axs = plot_sequence_and_prediction(
        observations=kwargs["observations"][: k + 2, :],
        actions=kwargs["actions"][: k + 1, :],
        tau=kwargs["env"].tau,
        obs_labels=kwargs["env"].obs_description,
        actions_labels=kwargs["env"].action_description,
        model=kwargs["model"],
        init_obs=kwargs["next_obs"],
        init_state=kwargs["next_state"],
        proposed_actions=kwargs["next_proposed_actions"],
    )

    plt.show()


def model_performance_callback(sequence_length: int = 10, batch_size: int = 5, show_plot: bool = True, **kwargs):
    """Callback that evaluates the model performance on a batch of data.

    Args:
        sequence_length (int): Length of the sequence to evaluate.
        batch_size (int): Number of sequences to evaluate.
        show_plot (bool): Whether to show the plot of the model performance.
        **kwargs: Keyword arguments passed to the callback.

    Returns:
        mses (float): Mean squared error of the model predictions averaged over the
            trajectories and the batch of trajectories.
    """
    starting_points, _ = precompute_starting_points(
        n_train_steps=1,
        k=kwargs["k"],
        sequence_length=sequence_length,
        training_batch_size=batch_size,
        loader_key=jax.random.key(seed=np.random.randint(low=100_000)),
    )

    batched_obs, batched_act = load_single_batch(
        observations_array=kwargs["observations"],
        actions_array=kwargs["actions"],
        starting_points=starting_points[0, ...],
        sequence_length=sequence_length,
    )

    mses = []

    model = kwargs["model"]
    env = kwargs["env"]

    for i in range(batched_obs.shape[0]):
        if show_plot:
            fig, _, pred_observations = plot_model_performance(
                model,
                batched_obs[i, ...],
                batched_act[i, ...],
                env.tau,
                env.obs_description,
                env.action_description,
            )
            plt.show()
        else:
            pred_observations = simulate_ahead(
                model,
                batched_obs[i, 0, :],
                batched_act[i, ...],
                env.tau,
            )

        mse = jnp.mean((pred_observations - batched_obs[i, ...]) ** 2)
        mses.append(mse)

    return np.mean(mses)


def optimization_prediction_accuracy_callback(**kwargs):
    """Callback that evaluates the accuracy of the prediction accuracy for the momentary
    iteration, i.e., explicitly for the actions that have been chosen based on the internal
    optimization loop. This is done by comparing the predicted observations with the actual
    observations and calculating the mean squared error.

    Currently, this callback is only implemented for two observation dimensions.
    """
    env = kwargs["env"]
    model = kwargs["model"]

    sim_actions = jnp.concatenate([kwargs["action"][None], kwargs["next_proposed_actions"][:-1]], axis=0)
    model_prediction = simulate_ahead(model, kwargs["obs"], sim_actions, env.tau)
    env_prediction, _ = simulate_ahead_with_env(env, kwargs["obs"], kwargs["state"], sim_actions)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(25, 6))

    axs[0].plot(model_prediction[..., 0], label="model")
    axs[0].plot(env_prediction[..., 0], label="env")

    axs[1].plot(model_prediction[..., 1], label="model")
    axs[1].plot(env_prediction[..., 1], label="env")

    axs[2].plot((model_prediction[..., 0] - env_prediction[..., 0]), "r", label=env.obs_description[0])
    axs[2].plot((model_prediction[..., 1] - env_prediction[..., 1]), "k", label=env.obs_description[1])

    for ax in axs:
        ax.grid(True)
        ax.legend()

    fig.suptitle("Optimization result accuracy")
    plt.show()
