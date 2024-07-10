import numpy as np


def simulate_ahead_with_env(env, obs, state, actions):
    observations = []
    observations.append(obs)

    for i in range(actions.shape[0]):
        obs, _, _, _, state = env.step(state, actions[i, :], env.env_properties)
        observations.append(obs)

    return np.vstack(observations), state
