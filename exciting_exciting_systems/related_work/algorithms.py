"""System excitation algorithms from related work."""

from typing import Callable

from tqdm.notebook import tqdm
import numpy as np
from cmaes import CMAwM

from pymoo.algorithms.soo.nonconvex.ga import GA

from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair

from exciting_exciting_systems.related_work.excitation_utils import (
    optimize_continuous_aprbs,
    optimize_permutation_aprbs,
    generate_aprbs,
    latin_hypercube_sampling,
)


def excite_with_GOATs(
    n_amplitudes: np.ndarray,
    env,
    bounds_duration: tuple,
    population_size: int,
    n_generations: int,
    n_support_points: int,
    featurize: Callable,
    seed: int = 0,
    verbose: bool = True,
):
    """System excitation using the GOATs algorithm from [Smits+Nelles2024].

    The optimization metric used here is MCUDSA as it is described in [Smits+Nelles2024]
    and it is optimized with a genetic algorithm.

    Args:
        n_amplitudes: The number of amplitudes to sample with Latin Hypercube Sampling
            and optimize w.r.t. the system excitation task
        env: The environment/system or model to excite
        bounds_duration: The upper and lower bound on the duration of a single amplitude
            as a tuple
        population_size: The number of individuals in the population of the GA
        n_generations: The number of generations of the GA
        n_support_points: The number of support point in MCUDSA
        featurize: Featurization of the observations before computation of the metric. If
            this is not necessary for the environment/system, pass the identity function
        seed: The seed for the genetic algorithm and LHS
        verbose: Whether the genetic algorithm print out its optimization progress

    Returns:
        observations: The observations gathered from the system
        actions: The actions applied to the system
    """

    obs, env_state = env.reset()
    obs = obs.astype(np.float32)
    env_state = env_state.astype(np.float32)

    opt_algorithm = GA(
        pop_size=population_size,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=1.0, eta=10.0, vtype=float, repair=RoundingRepair()),
        mutation=PM(prob=1.0, eta=10.0, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )

    observations, actions, _ = optimize_permutation_aprbs(
        opt_algorithm,
        amplitudes=latin_hypercube_sampling(d=env.action_space.shape[-1], n=n_amplitudes, seed=seed),
        env=env,
        obs=obs,
        env_state=env_state,
        bounds_duration=bounds_duration,
        n_generations=n_generations,
        support_points=latin_hypercube_sampling(
            d=(env.observation_space.shape[-1] + env.action_space.shape[-1]), n=n_support_points, seed=seed
        ),
        featurize=featurize,
        seed=seed,
        verbose=verbose,
        starting_observations=None,
        starting_actions=None,
    )

    return observations, actions


def excite_with_sGOATs(
    n_amplitudes: np.ndarray,
    n_amplitude_groups: int,
    reuse_observations: bool,
    all_observations: list,
    all_actions: list,
    env,
    bounds_duration: tuple,
    population_size: int,
    n_generations: int,
    n_support_points: int,
    featurize: Callable,
    seed=0,
    verbose=True,
):
    """System excitation using the sGOATs algorithm from [Smits+Nelles2024].

    The optimization metric used here is MCUDSA as it is described in [Smits+Nelles2024]
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
        n_support_points: The number of support point in MCUDSA
        featurize: Featurization of the observations before computation of the metric. If
            this is not necessary for the environment/system, pass the identity function
        seed: The seed for the genetic algorithm. TODO: This currently does not apply for
            the LHS which makes this kinda pointless to have a seed. To be fixed
        verbose: Whether the genetic algorithm print out its optimization progress

    Returns:
        all_observations: The finished observations list
        all_actions: The finished actions list
    """

    opt_algorithm = GA(
        pop_size=population_size,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=1.0, eta=10.0, vtype=float, repair=RoundingRepair()),
        mutation=PM(prob=1.0, eta=10.0, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )

    obs, env_state = env.reset()
    obs = obs.astype(np.float32)
    env_state = env_state.astype(np.float32)

    all_observations.append([obs[0]])

    all_amplitudes = latin_hypercube_sampling(d=env.action_space.shape[-1], n=n_amplitudes, seed=seed)
    amplitude_groups = np.split(all_amplitudes, n_amplitude_groups, axis=0)

    support_points = latin_hypercube_sampling(
        d=(env.observation_space.shape[-1] + env.action_space.shape[-1]), n=n_support_points, seed=seed
    )

    for amplitudes in amplitude_groups:

        # TODO: How big is the overhead of redefining the problem for each block in sGOATs?
        # TODO: The current implementation has x_0 in the starting observations twice? i think
        # so at least -> investigate
        observations, actions, last_env_state = optimize_permutation_aprbs(
            opt_algorithm=opt_algorithm,
            amplitudes=amplitudes,
            env=env,
            obs=obs,
            env_state=env_state,
            bounds_duration=bounds_duration,
            n_generations=n_generations,
            support_points=support_points,
            featurize=featurize,
            seed=seed,
            verbose=verbose,
            starting_observations=np.concatenate(all_observations) if reuse_observations else None,
            starting_actions=np.concatenate(all_actions) if reuse_observations else None,
        )

        # update obs and env_state as the starting point for the next amplitude group
        obs = observations[:, -1, :]
        env_state = last_env_state

        # save optimized actions and resulting observations
        all_observations.append(observations[0, 1:, :])
        all_actions.append(actions[0])

    return all_observations, all_actions


def excite_with_iGOATs(
    n_timesteps,
    env,
    actions,
    observations,
    h,
    a,  # TODO: implement the possiblity to not apply the full signal to the system
    bounds_amplitude,
    bounds_duration,
    population_size,
    n_generations,
    mean,
    sigma,
    featurize,
):
    """System excitation using the iGOATs algorithm from [Smits+Nelles2024]."""

    continuous_dim = h
    discrete_dim = h

    bounds = np.concatenate(
        [
            np.tile(bounds_amplitude, (continuous_dim, 1)),
            np.tile(bounds_duration, (discrete_dim, 1)),
        ]
    )
    steps = np.concatenate([np.zeros(continuous_dim), np.ones(discrete_dim)])

    obs, env_state = env.reset()
    obs = obs.astype(np.float32)
    env_state = env_state.astype(np.float32)

    observations.append(obs[0])

    pbar = tqdm(total=n_timesteps)
    while len(observations) < n_timesteps:
        optimizer = CMAwM(mean=mean, sigma=sigma, population_size=population_size, bounds=bounds, steps=steps)

        proposed_aprbs_params, values, optimizer = optimize_continuous_aprbs(
            optimizer,
            obs,
            env_state,
            np.stack(observations),
            n_generations=n_generations,
            env=env,
            h=h,
            featurize=featurize,
        )

        amplitudes = proposed_aprbs_params[:h]
        durations = proposed_aprbs_params[h:].astype(np.int32)

        new_actions = generate_aprbs(amplitudes=amplitudes, durations=durations)[None, :, None]

        # TODO: is this fair? The goal is to not go past the maximum number of steps
        # IMO needs to be reconsidered or discussed
        if new_actions.shape[1] + len(observations) > n_timesteps:
            new_actions = new_actions[:, : (n_timesteps - len(observations) + 1), :]

        for i in range(new_actions.shape[1]):
            action = new_actions[:, i, :]
            env_state = env.step(env_state, action)
            obs = env.generate_observation(env_state)

            observations.append(obs[0])
            actions.append(action[0])

        pbar.update(new_actions.shape[1])
    pbar.close()

    return observations, actions
