from typing import Callable

import numpy as np
from scipy.stats.qmc import LatinHypercube
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

from exciting_exciting_systems.related_work.np_reimpl.env_utils import simulate_ahead_with_env
from exciting_exciting_systems.related_work.np_reimpl.metrics import (
    MNNS_without_penalty,
    MC_uniform_sampling_distribution_approximation,
)


def latin_hypercube_sampling(d, n, seed=None):
    """Samples random points with latin hypercube sampling and normalizes between -1 and 1."""
    return LatinHypercube(d=d, seed=seed).random(n=n) * 2 - 1


def soft_penalty(a, a_max=1):
    """Computes penalty for the given input. Assumes symmetry in all dimensions."""
    relued_a = np.maximum(np.abs(a) - a_max, np.zeros(a.shape))
    penalty = np.sum(relued_a, axis=(-2, -1))
    return np.squeeze(penalty)


def generate_aprbs(amplitudes, durations):
    """Parameterizable aprbs. This is used to transform the aprbs parameters into a signal."""
    return np.concatenate([np.ones(duration) * amplitude for (amplitude, duration) in zip(amplitudes, durations)])


def fitness_function(env, obs, state, prev_observations, action_parameters, h, featurize):
    actions = generate_aprbs(amplitudes=action_parameters[:h], durations=action_parameters[h:].astype(np.int32))[
        :, None
    ]

    observations, _ = simulate_ahead_with_env(
        env,
        obs,
        state,
        actions,
    )
    feat_observations = featurize(observations)

    score = MNNS_without_penalty(data_points=featurize(prev_observations), new_data_points=feat_observations[1:, :])

    rho_obs = 1e10
    rho_act = 1e10
    penalty_terms = rho_obs * soft_penalty(a=observations, a_max=1) + rho_act * soft_penalty(a=actions, a_max=1)
    return np.squeeze(score).item() + penalty_terms.item()


def optimize_continuous_aprbs(optimizer, obs, env_state, prev_observations, n_generations, env, h, featurize):
    """Optimize an APRBS signal with chooseable amplitude levels for system excitiation."""
    for generation in range(n_generations):
        solutions = []
        x_for_eval_list = []

        for i in range(optimizer.population_size):
            x_for_eval, x_for_tell = optimizer.ask()
            value = fitness_function(env, obs, env_state, prev_observations, x_for_eval, h, featurize=featurize)

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


class GoatsProblem(ElementwiseProblem):
    """pymoo-API optimization problem for the GOATs and sGOATs algorithms.

    Optimizes amplitude permutations and durations of each specific amplitude.
    The amplitude levels are chosen beforehand.

    TODO: arbitrary observation and input dimensions
    """

    def __init__(
        self,
        amplitudes,
        env,
        obs,
        env_state,
        featurize,
        support_points,
        bounds_duration=(1, 50),
        starting_observations=None,
        starting_actions=None,
    ):

        n_amplitudes = amplitudes.shape[0]

        self.env = env
        self.obs = obs
        self.env_state = env_state
        self.featurize = featurize

        super().__init__(
            n_var=2 * n_amplitudes,
            n_obj=1,
            xl=np.concatenate([np.zeros(n_amplitudes), np.ones(n_amplitudes) * bounds_duration[0]]),
            xu=np.concatenate(
                [
                    np.ones(n_amplitudes) * np.linspace(0, n_amplitudes - 1, n_amplitudes)[::-1],
                    np.ones(n_amplitudes) * bounds_duration[1],
                ]
            ),
        )

        # The featurized support points
        self.support_points = featurize(support_points)

        self.amplitudes = amplitudes
        self.n_amplitudes = n_amplitudes
        if starting_observations is not None:
            self.starting_observations = featurize(starting_observations)
        else:
            self.starting_observations = None
        self.starting_actions = starting_actions

    @staticmethod
    def decode(lehmer_code: list[int]) -> list[int]:
        """Decode Lehmer code to permutation.

        This function decodes Lehmer code represented as a list of integers to a permutation.

        Source: https://optuna.readthedocs.io/en/latest/faq.html#how-can-i-deal-with-permutation-as-a-parameter
        """
        n = len(lehmer_code)

        all_indices = list(range(n))
        output = []
        for k in lehmer_code:
            value = all_indices[k]
            output.append(value)
            all_indices.remove(value)
        return output

    def _evaluate(self, x, out, *args, **kwargs):
        indices = self.decode(x[: self.n_amplitudes])

        applied_amplitudes = self.amplitudes[indices]

        actions = generate_aprbs(amplitudes=applied_amplitudes, durations=x[self.n_amplitudes :])[:, None]

        observations, _ = simulate_ahead_with_env(
            self.env,
            self.obs,
            self.env_state,
            actions,
        )

        feat_observations = self.featurize(observations)
        if self.starting_observations is not None:
            assert (
                self.starting_actions is not None
            ), "There are starting observations, but no corresponding starting actions!"
            feat_observations = np.concatenate([feat_observations, self.starting_observations])
            all_actions = np.concatenate([actions, self.starting_actions])
        else:
            all_actions = actions

        score = MC_uniform_sampling_distribution_approximation(
            data_points=np.concatenate([feat_observations[:-1, ...], all_actions], axis=-1),
            support_points=self.support_points,
        )
        N = observations.shape[0]

        rho_obs = 1
        rho_act = 1
        penalty_terms = rho_obs * soft_penalty(a=observations, a_max=1) + rho_act * soft_penalty(a=actions, a_max=1)

        out["F"] = 1 * score + penalty_terms.item()


def optimize_permutation_aprbs(
    opt_algorithm,
    amplitudes: np.ndarray,
    env,
    obs: np.ndarray,
    env_state: np.ndarray,
    bounds_duration: tuple,
    n_generations: int,
    support_points: np.ndarray,
    featurize: Callable,
    seed: int,
    verbose: bool,
    starting_observations: np.ndarray | None,
    starting_actions: np.ndarray | None,
):
    """Optimize an APRBS signal with predefined amplitude levels for system excitiation."""

    opt_problem = GoatsProblem(
        amplitudes=amplitudes,
        env=env,
        obs=obs,
        env_state=env_state,
        featurize=featurize,
        support_points=support_points,
        bounds_duration=bounds_duration,
        starting_observations=starting_observations,
        starting_actions=starting_actions,
    )

    res = minimize(
        problem=opt_problem,
        algorithm=opt_algorithm,
        termination=("n_gen", n_generations),
        seed=seed,
        save_history=False,
        verbose=verbose,
    )

    indices = opt_problem.decode(res.X[: opt_problem.n_amplitudes])
    applied_amplitudes = opt_problem.amplitudes[indices]
    applied_durations = res.X[opt_problem.n_amplitudes :]

    actions = generate_aprbs(amplitudes=applied_amplitudes, durations=applied_durations)[:, None]

    observations, last_env_state = simulate_ahead_with_env(
        env,
        obs,
        env_state,
        actions,
    )

    return observations, actions, last_env_state
