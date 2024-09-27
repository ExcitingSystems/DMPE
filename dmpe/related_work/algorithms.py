"""System excitation algorithms from related work."""

from typing import Callable

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pymoo.core.mixed import MixedVariableGA, MixedVariableDuplicateElimination

from dmpe.related_work.excitation_utils import (
    optimize_continuous_aprbs,
    optimize_permutation_aprbs,
    latin_hypercube_sampling,
)

import exciting_environments as exenvs
from dmpe.evaluation.plotting_utils import plot_sequence
from dmpe.related_work.mixed_GA import MixedVariableMating, MixedVariableSampling


def excite_with_GOATS(
    n_amplitudes: np.ndarray,
    env,
    bounds_duration: tuple,
    population_size: int,
    n_generations: int,
    featurize: Callable,
    rng: np.random.Generator,
    compress_data: bool,
    compression_target_N: int,
    rho_act: float,
    rho_obs: float,
    penalty_order: int,
    compression_dist_th: float,
    compression_feat_dim: int,
    verbose: bool = True,
):
    """System excitation using the GOATs algorithm from [Smits2024].

    The optimization metric used here is audze eglais as it is described in [Smits2024]
    and it is optimized with a genetic algorithm.

    Args:
        n_amplitudes: The number of amplitudes to sample with Latin Hypercube Sampling
            and optimize w.r.t. the system excitation task
        env: The environment/system or model to excite
        bounds_duration: The upper and lower bound on the duration of a single amplitude
            as a tuple
        population_size: The number of individuals in the population of the GA
        n_generations: The number of generations of the GA
        featurize: Featurization of the observations before computation of the metric. If
            this is not necessary for the environment/system, pass the identity function
        rng: The random number generator for the genetic algorithm and LHS
        verbose: Whether the genetic algorithm print out its optimization progress

    Returns:
        observations: The observations gathered from the system
        actions: The actions applied to the system
    """

    obs, env_state = env.reset()
    obs = obs.astype(np.float32)[0]
    if isinstance(env_state, np.ndarray):
        env_state = env_state.astype(np.float32)[0]

    opt_algorithm = MixedVariableGA(
        pop_size=population_size,
        sampling=MixedVariableSampling(),
        mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
    )

    observations, actions, _ = optimize_permutation_aprbs(
        opt_algorithm,
        amplitudes=latin_hypercube_sampling(d=env.action_dim, n=n_amplitudes, rng=rng),
        env=env,
        obs=obs,
        env_state=env_state,
        bounds_duration=bounds_duration,
        n_generations=n_generations,
        featurize=featurize,
        rng=rng,
        verbose=verbose,
        starting_observations=None,
        starting_actions=None,
        compress_data=compress_data,
        compression_target_N=compression_target_N,
        rho_act=rho_act,
        rho_obs=rho_obs,
        penalty_order=penalty_order,
        compression_dist_th=compression_dist_th,
        compression_feat_dim=compression_feat_dim,
        share_of_current_sequence=1,
    )

    return observations, actions


def generate_amplitude_groups(n_amplitudes: int, n_amplitude_groups: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate amplitude groups from a given number of amplitudes where each group in itself is
    "space-filling" in the sense that it is uniform on the interval [-1, 1].

    Args:
        n_amplitudes (int): The total number of amplitudes.
        n_amplitude_groups (int): The number of amplitude groups to generate.
        rng (np.random.Generator): The random number generator.

    Returns:
        amplitude_groups (np.ndarray): An array of amplitude groups, where each group contains a subset of amplitudes.

    Raises:
        AssertionError: If n_amplitudes is not divisible by n_amplitude_groups.

    """
    assert n_amplitudes % n_amplitude_groups == 0
    all_amplitudes = np.linspace(-1, 1, n_amplitudes)

    amplitude_groups = [[] for _ in range(n_amplitude_groups)]

    for idx, amplitude in enumerate(all_amplitudes):
        i = idx % n_amplitude_groups
        amplitude_groups[i].append(amplitude)

    rng.shuffle(amplitude_groups)
    amplitude_groups = np.array(amplitude_groups)

    for amplitude_group in amplitude_groups:
        rng.shuffle(amplitude_group)

    return amplitude_groups


def excite_with_sGOATS(
    n_amplitudes: np.ndarray,
    n_amplitude_groups: int,
    reuse_observations: bool,
    env: exenvs.CoreEnvironment,
    bounds_duration: tuple,
    population_size: int,
    n_generations: int,
    featurize: Callable,
    rng: np.random.Generator,
    compress_data: bool,
    compression_target_N: int,
    rho_act: float,
    rho_obs: float,
    penalty_order: int,
    compression_dist_th: float,
    compression_feat_dim: int,
    verbose: bool = True,
    plot_every_subsequence: bool = True,
):
    """System excitation using the sGOATs algorithm from [Smits2024].

    The optimization metric used here is MCUDSA as it is described in [Smits2024]
    and it is optimized with a genetic algorithm. It is similar to the GOATs algorithm
    with the major difference that the precomputed amplitude levels are not all used
    at once but only a subset of the amplitude levels is optimized at a time.

    Args:
        n_amplitudes: The number of amplitudes to sample with Latin Hypercube Sampling
            and optimize w.r.t. the system excitation task
        n_amplitude_groups: Decides in how many groups the amplitude levels are split
            for optimization
        reuse_observations: Whether to reuse the observations from previous amplitude
            groups in the metric computations of the following groups
        all_observations: A list to which the gathered observations are appended. When
            you want to use the algorithm as it is intended, this should be an empty
            list.
        all_actions: A list to which the applied actions are appended. When you want
            to use the algorithm as it is intended, this should be an empty list.
        env: The environment/system or model to excite
        bounds_duration: The upper and lower bound on the duration of a single amplitude
            as a tuple
        population_size: The number of individuals in the population of the GA
        n_generations: The number of generations of the GA
        featurize: Featurization of the observations before computation of the metric. If
            this is not necessary for the environment/system, pass the identity function
        rng: The rng for the genetic algorithm.
        verbose: Whether the genetic algorithm print out its optimization progress

    Returns:
        all_observations: The finished observations list
        all_actions: The finished actions list
    """

    all_observations = []
    all_actions = []

    opt_algorithm = MixedVariableGA(
        pop_size=population_size,
        sampling=MixedVariableSampling(),
        mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
    )

    obs, env_state = env.reset()
    obs = obs.astype(np.float32)[0]
    if isinstance(env_state, np.ndarray):
        env_state = env_state.astype(np.float32)[0]

    amplitude_groups = generate_amplitude_groups(
        n_amplitudes=n_amplitudes, n_amplitude_groups=n_amplitude_groups, rng=rng
    )

    for idx, amplitudes in enumerate(tqdm(amplitude_groups)):

        if len(all_observations) > 0 and reuse_observations:
            starting_observations = np.concatenate(all_observations)
        else:
            starting_observations = None

        if len(all_actions) > 0 and reuse_observations:
            starting_actions = np.concatenate(all_actions)
        else:
            starting_actions = None

        observations, actions, last_env_state = optimize_permutation_aprbs(
            opt_algorithm=opt_algorithm,
            amplitudes=amplitudes,
            env=env,
            obs=obs,
            env_state=env_state,
            bounds_duration=bounds_duration,
            n_generations=n_generations,
            featurize=featurize,
            rng=rng,
            verbose=verbose,
            starting_observations=starting_observations,
            starting_actions=starting_actions,
            compress_data=compress_data,
            compression_target_N=compression_target_N,
            rho_act=rho_act,
            rho_obs=rho_obs,
            penalty_order=penalty_order,
            compression_dist_th=compression_dist_th,
            compression_feat_dim=compression_feat_dim,
            share_of_current_sequence=1 / (idx + 1),
        )

        # update obs and env_state as the starting point for the next amplitude group
        obs = observations[-1, :]
        env_state = last_env_state

        # save optimized actions and resulting observations
        all_observations.append(observations[:-1, :])
        all_actions.append(actions)

        if plot_every_subsequence:
            fig, axs = plot_sequence(
                observations=np.concatenate(all_observations),
                actions=np.concatenate(all_actions)[:-1, ...],
                tau=env.tau,
                obs_labels=env.obs_description,
                action_labels=env.action_description,
            )
            plt.show()

    return all_observations, all_actions


def excite_with_iGOATS(
    n_timesteps,
    env,
    prediction_horizon,
    application_horizon,
    bounds_amplitude,
    bounds_duration,
    population_size,
    n_generations,
    featurize,
    rng,
    compress_data,
    compression_target_N,
    rho_obs,
    rho_act,
    penalty_order,
    compression_feat_dim,
    compression_dist_th,
    plot_subsequences=False,
):
    """System excitation using the iGOATs algorithm from [Smits2024]."""

    assert application_horizon <= prediction_horizon
    obs, env_state = env.reset()
    obs = obs.astype(np.float32)[0]
    if isinstance(env_state, np.ndarray):
        env_state = env_state.astype(np.float32)[0]

    all_actions = []
    all_observations = []

    opt_algorithm = MixedVariableGA(
        pop_size=population_size,
        sampling=MixedVariableSampling(),
        mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
    )

    pbar = tqdm(total=n_timesteps)
    while True:

        if len(all_observations) > 0 and len(all_actions) > 0:
            starting_observations = np.concatenate(all_observations)
            starting_actions = np.concatenate(all_actions)

        else:
            starting_observations = None
            starting_actions = None

        new_observations, new_actions, env_state = optimize_continuous_aprbs(
            opt_algorithm,
            prediction_horizon=prediction_horizon,
            application_horizon=application_horizon,
            env=env,
            obs=obs,
            env_state=env_state,
            bounds_amplitude=bounds_amplitude,
            bounds_duration=bounds_duration,
            n_generations=n_generations,
            featurize=featurize,
            rng=rng,
            starting_observations=starting_observations,
            starting_actions=starting_actions,
            compress_data=compress_data,
            compression_target_N=compression_target_N,
            rho_act=rho_act,
            rho_obs=rho_obs,
            penalty_order=penalty_order,
            compression_dist_th=compression_dist_th,
            compression_feat_dim=compression_feat_dim,
        )

        obs = new_observations[-1]

        all_observations.append(new_observations[:-1, :])
        all_actions.append(new_actions)

        if plot_subsequences:
            fig, axs = plot_sequence(
                observations=np.concatenate(all_observations),
                actions=np.concatenate(all_actions),
                tau=env.tau,
                obs_labels=env.obs_description,
                action_labels=env.action_description,
            )
            plt.show()

        pbar.update(new_actions.shape[0])

        if np.concatenate(all_observations).shape[0] >= n_timesteps:
            break

    pbar.close()

    return all_observations, all_actions
