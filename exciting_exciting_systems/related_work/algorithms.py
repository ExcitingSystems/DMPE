from tqdm.notebook import tqdm
import numpy as np
from scipy.stats.qmc import LatinHypercube
from cmaes import CMAwM

from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA

from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair

from exciting_exciting_systems.related_work.np_reimpl.env_utils import simulate_ahead_with_env
from exciting_exciting_systems.related_work.excitation_utils import optimize_aprbs, generate_aprbs, GoatsProblem


def excite_with_GOATs(
        n_amplitudes,
        env,
        bounds_duration,
        population_size,
        n_generations,
        n_support_points,
        featurize,
        seed=0,
        verbose=True
):
    
    obs, env_state = env.reset()
    obs = obs.astype(np.float32)
    env_state = env_state.astype(np.float32)

    # TODO: How big is the overhead of redefining the problem for each block in sGOATs?
    opt_problem = GoatsProblem(
        LatinHypercube(d=1).random(n=n_amplitudes) * 2 - 1,
        env,
        obs,
        env_state,
        featurize,
        bounds_duration,
        n_support_points,
    )

    opt_algorithm = GA(
        pop_size=population_size,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=1.0, eta=10.0, vtype=float, repair=RoundingRepair()),
        mutation=PM(prob=1.0, eta=10.0, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )

    res = minimize(
        opt_problem,
        opt_algorithm,
        termination=('n_gen', n_generations),
        seed=seed,
        save_history=False,
        verbose=verbose
    )

    indices = opt_problem.decode(res.X[:opt_problem.n_amplitudes])
    applied_amplitudes = opt_problem.amplitudes[indices]
    applied_durations = res.X[opt_problem.n_amplitudes:]

    actions = generate_aprbs(
        amplitudes=applied_amplitudes,
        durations=applied_durations
    )[None, :, None]

    observations, _ = simulate_ahead_with_env(
        env,
        obs,
        env_state,
        actions,
    )

    return observations, actions


def excite_with_iGOATs(
        n_timesteps,
        env,
        actions,
        observations,
        h,
        a,  # TODO: implement the possiblity to not use the full signal
        bounds_amplitude,
        bounds_duration,
        population_size,
        n_generations,
        mean,
        sigma,
        featurize
):
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
        optimizer = CMAwM(
            mean=mean,
            sigma=sigma,
            population_size=population_size,
            bounds=bounds,
            steps=steps
        )

        proposed_aprbs_params, values, optimizer = optimize_aprbs(
            optimizer,
            obs,
            env_state,
            np.stack(observations),
            n_generations=n_generations,
            env=env,
            h=h,
            featurize=featurize
        )

        amplitudes = proposed_aprbs_params[:h]
        durations = proposed_aprbs_params[h:].astype(np.int32)

        new_actions = generate_aprbs(
            amplitudes=amplitudes,
            durations=durations
        )[None, :, None]

        # TODO: is this fair? The goal is to not go past the maximum number of steps
        # IMO needs to be reconsidered or discusseds
        if new_actions.shape[1] + len(observations) > n_timesteps:
            new_actions = new_actions[: , :(n_timesteps - len(observations) + 1), :]

        for i in range(new_actions.shape[1]):
            action = new_actions[:, i, :]
            env_state = env.step(env_state, action)
            obs = env.generate_observation(env_state)

            observations.append(obs[0])
            actions.append(action[0])

        pbar.update(new_actions.shape[1])
    pbar.close()

    return observations, actions


def excite_with_sGOATs(
        n_amplitudes,
        n_amplitude_groups,
        all_observations,
        all_actions,
        env,
        bounds_duration,
        population_size,
        n_generations,
        n_support_points,
        featurize,
        seed=0,
        verbose=True
):
    
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
    
    all_amplitudes = LatinHypercube(d=1).random(n=n_amplitudes) * 2 - 1
    amplitude_groups = np.split(all_amplitudes, n_amplitude_groups, axis=0)

    for amplitudes in amplitude_groups:

        # TODO: How big is the overhead of redefining the problem for each block in sGOATs?
        opt_problem = GoatsProblem(
            amplitudes,
            env,
            obs,
            env_state,
            featurize,
            bounds_duration,
            n_support_points,
        )

        res = minimize(
            opt_problem,
            opt_algorithm,
            termination=('n_gen', n_generations),
            seed=seed,
            save_history=False,
            verbose=verbose
        )

        indices = opt_problem.decode(res.X[:opt_problem.n_amplitudes])
        applied_amplitudes = opt_problem.amplitudes[indices]
        applied_durations = res.X[opt_problem.n_amplitudes:]

        actions = generate_aprbs(
            amplitudes=applied_amplitudes,
            durations=applied_durations
        )[None, :, None]

        observations, last_env_state = simulate_ahead_with_env(
            env,
            obs,
            env_state,
            actions,
        )

        # update obs and env_state as the starting point for the next amplitude group
        obs = observations[:, -1, :]
        env_state = last_env_state

        # save optimized actions and resulting observations
        all_observations.append(observations[0, 1:, :])
        all_actions.append(actions[0])

    return all_observations, all_actions
