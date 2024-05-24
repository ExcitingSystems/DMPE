import numpy as np


def simulate_ahead_with_env(
       env,
       obs,
       state,
       actions 
):
    observations = []
    observations.append(obs)

    for i in range(actions.shape[1]):
        state = env.step(state, actions[:, i, :])
        obs = env.generate_observation(state)

        observations.append(obs)

    return np.stack(observations, axis=1)
