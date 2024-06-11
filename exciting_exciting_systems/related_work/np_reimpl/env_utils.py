import numpy as np


def simulate_ahead_with_env(env, obs, state, actions):
    observations = []
    observations.append(obs)

    state = state[None, ...]

    for i in range(actions.shape[0]):
        state = env.step(state, actions[None, i, :])
        obs = env.generate_observation(state)

        observations.append(obs[0])

    return np.vstack(observations), state[0]
