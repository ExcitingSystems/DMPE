import equinox as eqx


@eqx.filter_jit
def interact_and_observe(env, k, action, obs, state, actions, observations):
    """Interact with the environment and store the resulting effects."""

    # apply u_k = \hat{u}_{k+1} and go to x_{k+1}
    obs, _, _, _, state = env.step(state, action * env.env_properties.action_constraints.torque, env.env_properties)

    actions = actions.at[k].set(action)  # store u_k
    observations = observations.at[k + 1].set(obs)  # store x_{k+1}

    return obs, state, actions, observations
