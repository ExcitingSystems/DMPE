import equinox as eqx


@eqx.filter_jit
def interact_and_observe(
    env, k, action, obs, state, actions, observations
):
    """Interact with the environment and store the resulting effects."""
    
    # apply u_k = \hat{u}_{k+1} and go to x_{k+1}
    obs, _, _, _, state = env.step(action, state)

    actions = actions.at[k].set(action[0])  # store u_k
    observations = observations.at[k+1].set(obs[0])  # store x_{k+1}

    return obs, state, actions, observations
