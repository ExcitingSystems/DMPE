import numpy as np
import jax
import jax.numpy as jnp

from exciting_exciting_systems.models.model_utils import simulate_ahead_with_env
from exciting_exciting_systems.utils.metrics import MNNS_without_penalty
from exciting_exciting_systems.excitation.excitation_utils import soft_penalty


def generate_aprbs(amplitudes, durations):
    """Parameterizable aprbs. This is used to transform the aprbs parameters into a signal."""
    return np.concatenate([
        np.ones(duration) * amplitude for (amplitude, duration) in zip(amplitudes, durations)
    ])


def fitness_function(
        env,
        obs,
        state,
        prev_observations,
        action_parameters,
        h,
        max_duration,
        featurize
):
    actions = generate_aprbs(
        amplitudes=action_parameters[:h],
        durations=action_parameters[h:].astype(np.int32)
    )[:, None]

    max_signal_length = h * max_duration
    diff_to_max = max_signal_length - actions.shape[1]
    padded_actions = jnp.concatenate([actions, jnp.zeros((diff_to_max, 1))], axis=0)

    padded_observations = jax.vmap(simulate_ahead_with_env, in_axes=(None, 0, 0, 0, 0, 0, 0))(
        env,
        obs,
        state,
        padded_actions[None, ...],
        env.env_state_normalizer,
        env.action_normalizer,
        env.static_params
    )
    padded_observations = np.array(padded_observations[0])
    padded_feat_observations = featurize(padded_observations)
    feat_observations = padded_feat_observations[:-diff_to_max, :]

    score = MNNS_without_penalty(
        data_points=featurize(prev_observations),
        new_data_points=feat_observations[1:, :]
    )

    observations = padded_observations[:-diff_to_max, :]
    actions = padded_actions[:-diff_to_max, :]

    rho_obs = 1e10
    rho_act = 1e10
    penalty_terms = rho_obs * soft_penalty(a=observations, a_max=1) + rho_act * soft_penalty(a=actions, a_max=1)
    return jnp.squeeze(score).item() + penalty_terms.item()


def optimize_aprbs(
        optimizer,
        obs,
        env_state,
        prev_observations,
        n_generations,
        env,
        h,
        max_duration,
        featurize
):
    for generation in range(n_generations):
        solutions = []
        x_for_eval_list = []

        for i in range(optimizer.population_size):
            x_for_eval, x_for_tell = optimizer.ask()
            value = fitness_function(
                env,
                obs,
                env_state,
                prev_observations,
                x_for_eval,
                h,
                max_duration=max_duration,
                featurize=featurize
            )

            solutions.append((x_for_tell, value))
            x_for_eval_list.append(x_for_eval)

        optimizer.tell(solutions)

    values = []
    for x, value in solutions:
        values.append(value)

    xs = np.stack(x_for_eval_list)
    values = np.stack(values)
    min_idx = np.argmin(values)

    return xs[min_idx], values[min_idx], optimizer
