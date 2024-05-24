from tqdm.notebook import tqdm
import numpy as np
import jax.numpy as jnp

from cmaes import CMAwM

from exciting_exciting_systems.related_work.excitation_utils import optimize_aprbs, generate_aprbs


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
    obs = obs.astype(jnp.float32)
    env_state = env_state.astype(jnp.float32)

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
            jnp.stack(observations),
            n_generations=n_generations,
            env=env,
            h=h,
            max_duration=bounds[-1, -1],
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
            obs, _, _, _, env_state = env.step(action, env_state)

            observations.append(obs[0])
            actions.append(action[0])

        pbar.update(new_actions.shape[1])
    pbar.close()

    return observations, actions
