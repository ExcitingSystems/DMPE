from typing import Callable

import jax
import jax.numpy as jnp
import optax
import equinox as eqx

from exciting_environments.core_env import CoreEnvironment

from exciting_exciting_systems.models.model_utils import simulate_ahead, simulate_ahead_with_env
from exciting_exciting_systems.utils.density_estimation import (
    DensityEstimate, update_density_estimate, update_density_estimate_multiple_observations
)
from exciting_exciting_systems.utils.metrics import JSDLoss


@jax.jit
def soft_penalty(a, a_max=1):
    """Computes penalty for the given input. Assumes symmetry in all dimensions."""
    penalty = jnp.sum(jax.nn.relu(jnp.abs(a) - a_max), axis=(-2, -1))
    return jnp.squeeze(penalty)


@eqx.filter_jit
def loss_function(
        model,
        init_obs: jnp.ndarray,
        init_state: jnp.ndarray,
        actions: jnp.ndarray,
        density_estimate: DensityEstimate,
        tau: float,
        target_distribution: jnp.ndarray
) -> jnp.ndarray:
    """Predicts a trajectory based on the given actions and the model and computes the
    corresponding loss value.

    Args:
        model: The model to use for the prediction
        init_obs: The initial observation from which to start the simulation
        init_state: The initial state from which to start the simulation
        actions: The actions to apply in each step of the simulation, the length
            of the first dimension of this array determine the lenght of the
            output.
        density_estimate: The current estimate of the data density
        tau: The sampling time for the model
        target_distribution: The goal distribution of the data. The JSD loss is computed
            w.r.t. this distribution
    """

    if isinstance(model, CoreEnvironment):
        observations = simulate_ahead_with_env(
            env=model,
            init_obs=init_obs,
            init_state=init_state,
            actions=actions,
            env_state_normalizer=model.env_state_normalizer[0, :],
            action_normalizer=model.action_normalizer[0, :],
            static_params={key: value[0, :] for (key, value) in model.static_params.items()}
        )
    else:
        observations = simulate_ahead(
            model=model,
            init_obs=init_obs,
            actions=actions,
            tau=tau
        )

    predicted_density_estimate = update_density_estimate_multiple_observations(density_estimate, observations)
    loss = JSDLoss(
        p=predicted_density_estimate.p,
        q=target_distribution
    )

    # TODO: pull this automatically, maybe penalty_kwargs or something
    rho_obs = 1e4
    rho_act = 1e4
    penalty_terms = rho_obs * soft_penalty(a=observations, a_max=1) + rho_act * soft_penalty(a=actions, a_max=1)

    return loss + penalty_terms


@eqx.filter_jit
def optimize_actions(
        grad_loss_function,
        proposed_actions,
        model,
        solver,
        init_obs,
        init_state,
        density_estimate,
        tau,
        target_distribution
):
    """Uses the model to compute the effect of actions onto the state/observation trajectory to 
    optimize the actions w.r.t. the given (gradient of the) loss function.
    """

    opt_state = solver.init(proposed_actions)

    def body_fun(i, carry):
        proposed_actions, opt_state = carry
        grad = jax.vmap(
            grad_loss_function,
            in_axes=(None, 0, 0, 0, DensityEstimate(0, None, None, None), None, None),
        )(
            model,
            init_obs,
            init_state,
            proposed_actions,
            density_estimate,
            tau,
            target_distribution
        )
        updates, opt_state = solver.update(grad, opt_state, proposed_actions)
        proposed_actions = optax.apply_updates(proposed_actions, updates)
        return (proposed_actions, opt_state)

    proposed_actions, _ = jax.lax.fori_loop(0, 5, body_fun, (proposed_actions, opt_state))
    return proposed_actions


class Exciter(eqx.Module):
    """A class that carries the necessary tools for excitation input computations.
    
    Args:
        grad_loss_function: The gradient of the loss function w.r.t. the actions as
            a callable function
        excitiation_solver: The solver/optimizer for the excitation input computation
        tau: The time step length of the simulation
        target_distribution: The targeted distribution for the data density
    """

    grad_loss_function: Callable
    excitation_solver: optax._src.base.GradientTransformationExtraArgs
    tau: float
    target_distribution: jnp.ndarray

    @eqx.filter_jit
    def choose_action(
            self,
            obs: jnp.ndarray,
            state: jnp.ndarray,
            model,
            density_estimate: DensityEstimate,
            proposed_actions: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, DensityEstimate]:
        """Chooses the next action to take, updates the density estimate and 
        proposes future actions.

        Args:
            obs: The current observations from which to start
            state: The current state from which to start
            model: The current model of the environment used for the prediction
            density_estimate: The current estimate of the data density without
                the current step k
            propsed_actions: An initial proposition of actions to take

        Returns:
            action: The chosen action
            next_proposed_actions: An initial proposition for future actions
            density_estimate: The updated density estimate now incorporating
                the current step k
        """
        proposed_actions = optimize_actions(
            grad_loss_function=self.grad_loss_function,
            proposed_actions=proposed_actions,
            model=model,
            solver=self.excitation_solver,
            init_obs=obs,
            init_state=state,
            density_estimate=density_estimate,
            tau=self.tau,
            target_distribution=self.target_distribution
        )
    
        # update grid KDE with x_k
        density_estimate = jax.vmap(
            update_density_estimate,
            in_axes=[DensityEstimate(0, None, None, None), 0],
            out_axes=DensityEstimate(0, None, None, None)
        )(density_estimate, obs)
    
        action = proposed_actions[:, 0, :]
        next_proposed_actions = proposed_actions.at[:, :-1, :].set(proposed_actions[:, 1:, :])
    
        return action, next_proposed_actions, density_estimate
