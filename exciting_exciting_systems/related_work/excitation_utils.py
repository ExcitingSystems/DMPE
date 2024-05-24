import numpy as np

from exciting_exciting_systems.related_work.np_reimpl.env_utils import simulate_ahead_with_env
from exciting_exciting_systems.related_work.np_reimpl.metrics import MNNS_without_penalty


def soft_penalty(a, a_max=1):
    """Computes penalty for the given input. Assumes symmetry in all dimensions.
    """
    relued_a = np.maximum(np.abs(a) - a_max, np.zeros(a.shape))
    penalty = np.sum(relued_a, axis=(-2, -1))
    return np.squeeze(penalty)


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
        featurize
):
    actions = generate_aprbs(
        amplitudes=action_parameters[:h],
        durations=action_parameters[h:].astype(np.int32)
    )[:, None]

    observations = simulate_ahead_with_env(
        env,
        obs,
        state,
        actions[None, ...],
    )
    feat_observations = featurize(observations)

    score = MNNS_without_penalty(
        data_points=featurize(prev_observations),
        new_data_points=feat_observations[0, 1:, :]
    )

    rho_obs = 1e10
    rho_act = 1e10
    penalty_terms = rho_obs * soft_penalty(a=observations, a_max=1) + rho_act * soft_penalty(a=actions, a_max=1)
    return np.squeeze(score).item() + penalty_terms.item()


def optimize_aprbs(
        optimizer,
        obs,
        env_state,
        prev_observations,
        n_generations,
        env,
        h,
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
