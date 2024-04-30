import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import jax
import jax.numpy as jnp
import equinox as eqx

from exciting_exciting_systems.evaluation.plotting_utils import plot_sequence_and_prediction
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
    init_opt_state
):
    """Fit the model on the gathered data."""

    dynamic_init_model_state, static_model_state = eqx.partition(model, eqx.is_array)
    init_carry = (dynamic_init_model_state, init_opt_state)

    def body_fun(i, carry):
        dynamic_model_state, opt_state = carry
        model_state = eqx.combine(static_model_state, dynamic_model_state)

        batched_observations, batched_actions = load_single_batch(
            observations, actions, starting_points[i, ...], sequence_length
        )
        new_model_state, new_opt_state = make_step(
            model_state,
            batched_observations,
            batched_actions,
            tau,
            opt_state,
            featurize,
            optim
        )

        new_dynamic_model_state, new_static_model_state = eqx.partition(new_model_state, eqx.is_array)
        assert eqx.tree_equal(static_model_state, new_static_model_state) is True
        return (new_dynamic_model_state, new_opt_state)

    final_dynamic_model_state, final_opt_state = jax.lax.fori_loop(lower=0, upper=n_train_steps,body_fun=body_fun, init_val=init_carry)
    final_model = eqx.combine(static_model_state, final_dynamic_model_state)
    return final_model, final_opt_state


@eqx.filter_jit
def precompute_starting_points(
        n_train_steps,
        k,
        sequence_length,
        training_batch_size,
        loader_key
):
    index_normalized = jax.random.uniform(loader_key, shape=(n_train_steps, training_batch_size)) * (k + 1 - sequence_length)
    starting_points = index_normalized.astype(jnp.int32) 
    (loader_key,) = jax.random.split(loader_key, 1)

    return starting_points, loader_key


def excite_and_fit(
        n_timesteps,
        env,
        grad_loss_function,
        obs,
        state,
        proposed_actions,
        p_est,
        x_g,
        bandwidth,
        tau,
        target_distribution,
        n_prediction_steps,
        training_batch_size,
        model,
        n_train_steps,
        sequence_length,
        observations,
        actions,
        featurize,
        solver_prediction,
        solver_model,
        opt_state_model,
        loader_key
):
    """Main algorithm to throw at a given (unknown) system and generate informative data from that system.

    Args:

    Returns:

    """
    for k in tqdm(range(n_timesteps)):
        obs, state, actions, observations, proposed_actions, p_est = excite(
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
            jnp.array([k]),
            bandwidth,
            tau,
            target_distribution
        )

        if k > n_prediction_steps:
            starting_points, loader_key = precompute_starting_points(
                n_train_steps, jnp.array([k]), sequence_length, training_batch_size, loader_key
            )

            model, opt_state_model = fit(
                model,
                n_train_steps,
                starting_points,
                sequence_length,
                observations, 
                actions,
                tau,
                featurize,
                solver_model,
                opt_state_model
            )

        # if k % 500 == 0 and k > 0:
        #     fig, axs = plot_sequence_and_prediction(
        #         observations=observations[:k+2,:],
        #         actions=actions[:k+1,:],
        #         tau=tau,
        #         obs_labels=[r"$\theta$", r"$\omega$"],
        #         actions_labels=[r"$u$"],
        #         model=env,
        #         init_obs=obs[0, :],
        #         init_state=state[0, :],
        #         proposed_actions=proposed_actions[0, :]
        #     )
        #     plt.show()

    return observations, actions, model
