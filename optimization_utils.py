from functools import partial

from tqdm.notebook import tqdm

import jax
import jax.numpy as jnp
import optax

from model_utils import simulate_ahead
from density_estimation import update_kde_grid_multiple_observations
from metrics import JSDLoss


@partial(jax.jit, static_argnums=(1, 4))
def loss_function(
        actions,
        model,
        init_obs,
        init_state,
        n_steps,
        p_est,
        x,
        start_n_measurments,
        bandwidth,
        target_distribution
    ):

    actions = jax.nn.tanh(actions)  # TODO: sketchy
    observations = simulate_ahead(
        model=model,
        n_steps=n_steps,
        obs=init_obs,
        state=init_state,
        actions=actions
    )

    p_est = update_kde_grid_multiple_observations(p_est, x, observations, start_n_measurments, bandwidth)
    loss = JSDLoss(
        p=p_est,
        q=target_distribution
    )
    return loss


def optimize(
        grad_loss_function,
        proposed_actions,
        model,
        init_obs,
        init_state,
        n_steps,
        p_est,
        x,
        start_n_measurments,
        bandwidth,
        target_distribution
    ):

    solver = optax.adabelief(learning_rate=1e-1)
    opt_state = solver.init(proposed_actions)

    for iter in range(100):
        grad = grad_loss_function(
            proposed_actions,
            model,
            init_obs,
            init_state,
            n_steps,
            p_est,
            x,
            start_n_measurments,
            bandwidth,
            target_distribution
        )
        updates, opt_state = solver.update(grad, opt_state, proposed_actions)
        proposed_actions = optax.apply_updates(proposed_actions, updates)

    return proposed_actions
