import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import jax
import jax.numpy as jnp
import equinox as eqx

from exciting_exciting_systems.algorithms.algorithm_utils import interact_and_observe
from exciting_exciting_systems.models.model_training import precompute_starting_points, fit
from exciting_exciting_systems.evaluation.plotting_utils import plot_sequence_and_prediction


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
):
    """Main algorithm to throw at a given (unknown) system and generate informative data from that system.

    Args:

    Returns:

    """
    for k in tqdm(range(n_timesteps)):
        action, proposed_actions, density_estimate = exciter.choose_action(
            obs=obs, state=state, model=model, density_estimate=density_estimate, proposed_actions=proposed_actions
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

        if k % 500 == 0 and k > 0:
            fig, axs = plot_sequence_and_prediction(
                observations=observations[: k + 2, :],
                actions=actions[: k + 1, :],
                tau=exciter.tau,
                obs_labels=[r"$\theta$", r"$\omega$"],
                actions_labels=[r"$u$"],
                model=model,
                init_obs=obs[0, :],
                init_state=state[0, :],
                proposed_actions=proposed_actions[0, :],
            )
            plt.show()

    return observations, actions, model, density_estimate
