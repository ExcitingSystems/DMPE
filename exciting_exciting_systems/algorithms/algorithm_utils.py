import equinox as eqx


@eqx.filter_jit
def interact_and_observe(env, k, action, state, actions, observations):
    """
    Interact with the environment and store the resulting effects.

    Args:
        env: The environment object.
        k: The current time step.
        action: The action to be taken at time step k.
        state: The state of the environment at time step k.
        actions: The list of actions taken so far.
        observations: The list of observations observed so far.

    Returns:
        obs: The updated observation at time step k+1.
        state: The updated state of the environment at time step k+1.
        actions: The updated list of actions taken so far.
        observations: The updated list of observations observed so far.
    """

    # apply u_k = \hat{u}_{k+1} and go to x_{k+1}
    obs, _, _, _, state = env.step(state, action, env.env_properties)

    actions = actions.at[k].set(action)  # store u_k
    observations = observations.at[k + 1].set(obs)  # store x_{k+1}

    return obs, state, actions, observations
