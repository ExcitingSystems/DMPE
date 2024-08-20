from operator import itemgetter
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.qmc import LatinHypercube
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

import exciting_exciting_systems
from exciting_exciting_systems.related_work.np_reimpl.env_utils import simulate_ahead_with_env
from exciting_exciting_systems.related_work.np_reimpl.metrics import (
    MNNS_without_penalty,
    audze_eglais,
)
from exciting_exciting_systems.related_work.mixed_GA import Permutation, Integer, Real

import exciting_environments as excenvs


def latin_hypercube_sampling(d, n, rng):
    """Samples random points with latin hypercube sampling and normalizes between -1 and 1."""
    return LatinHypercube(d=d, seed=rng).random(n=n) * 2 - 1


def soft_penalty(a, a_max=1, penalty_order=2):
    """Computes penalty for the given input. Assumes symmetry in all dimensions."""
    relued_a = np.maximum(np.abs(a) - a_max, np.zeros(a.shape))

    relued_a = relued_a**penalty_order

    penalty = np.sum(relued_a, axis=(-2, -1))
    return np.squeeze(penalty)


def generate_aprbs(amplitudes, durations):
    """Parameterizable aprbs. This is used to transform the aprbs parameters into a signal."""
    return np.concatenate([np.ones(duration) * amplitude for (amplitude, duration) in zip(amplitudes, durations)])


def compress_datapoints(datapoints, N_c, feature_dimension, dist_th):
    """
    Compresses a sequence of datapoints based for the GOATS algorthims.

    Args:
        datapoints (ndarray): The sequence of datapoints to be compressed.
        N_c (int): The number of extra points per sequence.
        feature_dimension (int): The index of the feature dimension in the datapoints array.
        dist_th (float): The threshold distance for determining the number of extra points.

    Returns:
        compressed_data (ndarray): The compressed sequence of datapoints.
        indices (list): The indices of the selected datapoints in the original sequence.

    """
    # split data
    considered_data = datapoints[..., feature_dimension]
    support_mask = np.diff(np.sign(considered_data)) != 0
    support_mask = np.concatenate([support_mask, np.ones(1, dtype=bool)], axis=0)

    curv_approx = np.diff(considered_data)
    curv_support_mask = np.diff(np.sign(curv_approx)) != 0
    curv_support_mask = np.concatenate([curv_support_mask, np.ones(2, dtype=bool)], axis=0)
    support_mask = np.logical_or(curv_support_mask, support_mask)

    support = considered_data[support_mask]
    support_points = datapoints[support_mask]

    # compute number of extra points per subsequence
    support_distances = np.concatenate([np.diff(support), np.zeros(1)])
    support_abs_distances = np.abs(support_distances)
    n_per_subsequence = np.zeros(support_distances.shape)

    n_per_subsequence[support_abs_distances > dist_th] = support_abs_distances[support_abs_distances > dist_th]
    n_per_subsequence *= N_c / np.sum(n_per_subsequence)
    n_per_subsequence = n_per_subsequence.astype(np.int32)

    # bring data together
    support_indices = np.where(support_mask == 1)[0]

    compressed_data = []
    indices = []

    for idx, (start, start_point, n_new_points, distance) in enumerate(
        zip(support, support_points, n_per_subsequence, support_distances)
    ):
        compressed_data.append(start_point)
        indices.append(support_indices[idx])

        if idx + 1 >= support.shape[0]:
            available_support_indices = np.arange(support_indices[idx] + 1, support.shape[0])
        else:
            available_support_indices = np.arange(support_indices[idx] + 1, support_indices[idx + 1])
        available_support = considered_data[available_support_indices]

        if n_new_points > 0 and len(available_support) > 0:
            new_samples = np.linspace(start, distance + start, n_new_points + 2)[1:-1]
            for sample in new_samples:
                dist = np.abs(sample - available_support)
                chosen_idx = np.argmin(dist)
                chosen_obs = datapoints[available_support_indices[chosen_idx]]
                if chosen_idx not in indices:
                    compressed_data.append(chosen_obs)
                    indices.append(available_support_indices[chosen_idx])

    compressed_data = np.stack(compressed_data)

    return compressed_data, indices


class ContinuousGoatsProblem(ElementwiseProblem):
    """pymoo-API optimization problem for the iGOATs algorithm.

    Optimizes amplitude permutations and durations of each specific amplitude.
    The amplitude levels are chosen beforehand.
    """

    def __init__(
        self,
        prediction_horizon,
        env,
        obs,
        env_state,
        featurize,
        bounds_amplitude,
        bounds_duration,
        starting_observations,
        starting_actions,
        compress_data,
        compression_target_N,
        rho_obs,
        rho_act,
        penalty_order,
        compression_feat_dim,
        compression_dist_th,
    ):

        self.env = env
        self.obs = obs
        self.env_state = env_state
        self.featurize = featurize

        amplitude_variables = {f"a_{number}": Real(bounds=bounds_amplitude) for number in range(prediction_horizon)}
        duration_variables = {f"d_{number}": Integer(bounds=bounds_duration) for number in range(prediction_horizon)}
        all_vars = dict(amplitude_variables, **duration_variables)

        self.permutation_keys = tuple()
        self.non_permutation_keys = tuple(all_vars.keys())

        super().__init__(
            vars=all_vars,
            n_obj=1,
        )

        self.prediction_horizon = prediction_horizon

        if len(starting_observations) > 0 and len(starting_actions) > 0:
            starting_observations = featurize(np.stack(starting_observations))
            starting_actions = np.stack(starting_actions)

            self.starting_feat_datapoints = np.concatenate([starting_observations, starting_actions], axis=-1)
            if compress_data and len(starting_observations) > 2000:
                self.starting_feat_datapoints, _ = compress_datapoints(
                    self.starting_feat_datapoints,
                    N_c=int(compression_target_N),
                    feature_dimension=compression_feat_dim,
                    dist_th=compression_dist_th,
                )

        else:
            self.starting_feat_datapoints = None

        self.compress_data = compress_data
        self.compression_target_N = compression_target_N
        self.rho_obs = rho_obs
        self.rho_act = rho_act
        self.penalty_order = penalty_order
        self.compression_feat_dim = compression_feat_dim
        self.compression_dist_th = compression_dist_th

    def _evaluate(self, x, out, *args, **kwargs):
        action_parameters = np.fromiter(x.values(), dtype=np.float64)
        actions = generate_aprbs(
            amplitudes=action_parameters[: self.prediction_horizon],
            durations=action_parameters[self.prediction_horizon :].astype(np.int32),
        )[:, None]

        observations, _ = simulate_ahead_with_env(
            self.env,
            self.obs,
            self.env_state,
            actions,
        )
        feat_observations = self.featurize(observations)
        new_datapoints = np.concatenate([feat_observations[:-1], actions], axis=-1)

        if self.starting_feat_datapoints is None:
            score = audze_eglais(new_datapoints)
        else:
            score = MNNS_without_penalty(
                data_points=self.starting_feat_datapoints,
                new_data_points=new_datapoints,
            )

        penalty_terms = self.rho_obs * soft_penalty(
            a=observations, a_max=1, penalty_order=self.penalty_order
        ) + self.rho_act * soft_penalty(a=actions, a_max=1, penalty_order=self.penalty_order)
        out["F"] = np.squeeze(score).item() + penalty_terms.item()


def optimize_continuous_aprbs(
    opt_algorithm,
    prediction_horizon: int,
    application_horizon: int,
    env,
    obs: np.ndarray,
    env_state: np.ndarray,
    bounds_amplitude: tuple,
    bounds_duration: tuple,
    n_generations: int,
    featurize: Callable,
    rng: np.random.Generator,
    starting_observations: np.ndarray,
    starting_actions: np.ndarray,
    compress_data: bool,
    compression_target_N: int,
    rho_obs: float,
    rho_act: float,
    penalty_order: int,
    compression_feat_dim: int,
    compression_dist_th: float,
):
    """Optimize an APRBS signal with continuous amplitude levels for system excitiation."""

    opt_problem = ContinuousGoatsProblem(
        prediction_horizon,
        env,
        obs,
        env_state,
        featurize,
        bounds_amplitude,
        bounds_duration,
        starting_observations=starting_observations,
        starting_actions=starting_actions,
        compress_data=compress_data,
        compression_target_N=compression_target_N,
        rho_obs=rho_obs,
        rho_act=rho_act,
        penalty_order=penalty_order,
        compression_feat_dim=compression_feat_dim,
        compression_dist_th=compression_dist_th,
    )

    res = minimize(
        problem=opt_problem,
        algorithm=opt_algorithm,
        termination=("n_gen", n_generations),
        seed=rng.integers(low=0, high=2**32 - 1, size=1).item(),
        save_history=False,
        verbose=False,
    )
    proposed_aprbs_params = np.fromiter(res.X.values(), dtype=np.float64)

    amplitudes = proposed_aprbs_params[:application_horizon]
    all_durations = proposed_aprbs_params[prediction_horizon:]
    durations = all_durations[:application_horizon].astype(np.int32)
    new_actions = generate_aprbs(amplitudes=amplitudes, durations=durations)[:, None]

    new_observations, env_state = simulate_ahead_with_env(
        env,
        obs,
        env_state,
        new_actions,
    )
    return new_observations, new_actions, env_state


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
        bounds_duration=(1, 50),
        starting_observations=None,
        starting_actions=None,
        compress_data: bool = True,
        compression_target_N: int = 500,
        rho_obs: float = 1e3,
        rho_act: float = 1e3,
        penalty_order: int = 2,
        compression_feat_dim: int = 0,
        compression_dist_th: float = 0.1,
        share_of_current_sequence: float = 1,
    ):

        n_amplitudes = amplitudes.shape[0]

        self.env = env
        self.obs = obs
        self.env_state = env_state
        self.featurize = featurize

        amplitude_variables = {f"a_{number}": Permutation(bounds=(0, n_amplitudes)) for number in range(n_amplitudes)}
        duration_variables = {f"d_{number}": Integer(bounds=bounds_duration) for number in range(n_amplitudes)}

        self.permutation_keys = tuple(amplitude_variables.keys())
        self.non_permutation_keys = tuple(duration_variables.keys())

        all_vars = dict(amplitude_variables, **duration_variables)

        super().__init__(
            vars=all_vars,
            n_obj=1,
        )

        self.amplitudes = amplitudes
        self.n_amplitudes = n_amplitudes
        if starting_observations is not None and starting_actions is not None:
            starting_observations = featurize(starting_observations)
            starting_actions = starting_actions

            self.starting_feat_datapoints = np.concatenate([starting_observations, starting_actions], axis=-1)
            if compress_data and len(self.starting_feat_datapoints) > 2000:
                self.starting_feat_datapoints, _ = compress_datapoints(
                    self.starting_feat_datapoints,
                    N_c=int(compression_target_N * (1 - share_of_current_sequence)),
                    feature_dimension=compression_feat_dim,
                    dist_th=compression_dist_th,
                )
        else:
            self.starting_feat_datapoints = None

        self.compress_data = compress_data
        self.compression_target_N = compression_target_N
        self.rho_obs = rho_obs
        self.rho_act = rho_act
        self.penalty_order = penalty_order
        self.compression_feat_dim = compression_feat_dim
        self.compression_dist_th = compression_dist_th
        self.share_of_current_sequence = share_of_current_sequence

    def _evaluate(self, x, out, *args, **kwargs):
        indices = np.array(itemgetter(*self.permutation_keys)(x))
        durations = np.array(itemgetter(*self.non_permutation_keys)(x))

        applied_amplitudes = self.amplitudes[indices]

        actions = generate_aprbs(amplitudes=applied_amplitudes, durations=durations)[:, None]

        if isinstance(self.env, excenvs.core_env.CoreEnvironment):
            observations, _ = exciting_exciting_systems.models.model_utils.simulate_ahead_with_env(
                self.env,
                self.obs,
                self.env_state,
                actions,
            )
        else:
            observations, _ = simulate_ahead_with_env(
                self.env,
                self.obs,
                self.env_state,
                actions,
            )

        feat_observations = self.featurize(observations)
        feat_datapoints = np.concatenate([feat_observations[:-1], actions], axis=-1)

        if self.compress_data:
            feat_datapoints, indices = compress_datapoints(
                feat_datapoints,
                N_c=int(self.compression_target_N * self.share_of_current_sequence),
                feature_dimension=self.compression_feat_dim,
                dist_th=self.compression_dist_th,
            )

        if self.starting_feat_datapoints is not None:
            feat_datapoints = np.concatenate([self.starting_feat_datapoints, feat_datapoints])

        score = audze_eglais(feat_datapoints)

        # TODO: should the number of steps be included in the loss? This is to incite shorter trajectories..?
        N = feat_datapoints.shape[0]

        penalty_terms = self.rho_obs * soft_penalty(
            a=observations, a_max=1, penalty_order=self.penalty_order
        ) + self.rho_act * soft_penalty(a=actions, a_max=1, penalty_order=self.penalty_order)

        out["F"] = N * score + penalty_terms.item()


def optimize_permutation_aprbs(
    opt_algorithm,
    amplitudes: np.ndarray,
    env,
    obs: np.ndarray,
    env_state: np.ndarray,
    bounds_duration: tuple,
    n_generations: int,
    featurize: Callable,
    rng: np.random.Generator,
    verbose: bool,
    starting_observations: np.ndarray | None,
    starting_actions: np.ndarray | None,
    compress_data: bool,
    compression_target_N: int,
    rho_obs: float,
    rho_act: float,
    penalty_order: int,
    compression_feat_dim: int,
    compression_dist_th: float,
    share_of_current_sequence: float,
):
    """Optimize an APRBS signal with predefined amplitude levels for system excitiation."""

    opt_problem = GoatsProblem(
        amplitudes=amplitudes,
        env=env,
        obs=obs,
        env_state=env_state,
        featurize=featurize,
        bounds_duration=bounds_duration,
        starting_observations=starting_observations,
        starting_actions=starting_actions,
        compress_data=compress_data,
        compression_target_N=compression_target_N,
        rho_act=rho_act,
        rho_obs=rho_obs,
        penalty_order=penalty_order,
        compression_dist_th=compression_dist_th,
        compression_feat_dim=compression_feat_dim,
        share_of_current_sequence=share_of_current_sequence,
    )

    res = minimize(
        problem=opt_problem,
        algorithm=opt_algorithm,
        termination=("n_gen", n_generations),
        seed=rng.integers(low=0, high=2**32 - 1, size=1).item(),
        save_history=False,
        verbose=verbose,
    )

    indices = np.array(itemgetter(*opt_problem.permutation_keys)(res.X))
    applied_amplitudes = opt_problem.amplitudes[indices]

    applied_durations = np.array(itemgetter(*opt_problem.non_permutation_keys)(res.X))

    actions = generate_aprbs(amplitudes=applied_amplitudes, durations=applied_durations)[:, None]

    observations, last_env_state = simulate_ahead_with_env(
        env,
        obs,
        env_state,
        actions,
    )

    return observations, actions, last_env_state
