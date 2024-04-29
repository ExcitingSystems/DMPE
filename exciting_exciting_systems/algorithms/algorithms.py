import jax
import jax.numpy as jnp
import equinox as eqx


from exciting_exciting_systems.optimization import choose_action
from exciting_exciting_systems.models.model_training import load_single_batch, make_step


@eqx.filter_jit
def excite(
    env,
    actions,
    observations,
    grad_loss_function,
    proposed_actions,
    model,
    solver_prediction,
    obs,
    state,
    p_est,
    x_g,
    k,
    bandwidth,
    tau,
    target_distribution

):
    """Choose an action and apply it on the system.
    
    Only jit-compilable if the call to the environment's step function is jit-compilable.   
    """

    action, proposed_actions, p_est = choose_action(
        grad_loss_function,
        proposed_actions,
        model,
        solver_prediction,
        obs,
        state,
        p_est,
        x_g,
        k,
        bandwidth,
        tau,
        target_distribution
    )

    # apply u_k = \hat{u}_{k+1} and go to x_{k+1}
    obs, _, _, _, state = env.step(action, state)

    actions = actions.at[k].set(action[0])  # store u_k
    observations = observations.at[k+1].set(obs[0])  # store x_{k+1}

    return obs, state, actions, observations, proposed_actions, p_est


@eqx.filter_jit
@eqx.debug.assert_max_traces(max_traces=1)
def fit(
    model,
    n_train_steps,
    starting_points,
    sequence_length,
    observations,
    actions,
    tau,
    featurize,
    optim,
    opt_state
):
    """Fit the model on the gathered data."""
    for (i, iter_starting_points) in zip(range(n_train_steps), starting_points):

        batched_observations, batched_actions = load_single_batch(
            observations, actions, iter_starting_points, sequence_length
        )
        model_training_loss, model, opt_state = make_step(
            model,
            batched_observations,
            batched_actions,
            tau,
            opt_state,
            featurize,
            optim
        )

    return model_training_loss, model, opt_state


@eqx.filter_jit
def excite_and_fit(
        
):
    """Main algorithm to throw at a given (unknown) system and generate informative data from that system.
    
    Args:

    Returns:

    """