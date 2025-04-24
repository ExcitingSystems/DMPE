from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp
import optax
import equinox as eqx

import exciting_environments as excenvs

from dmpe.models.model_utils import simulate_ahead, simulate_ahead_with_env
from dmpe.utils.density_estimation import (
    DensityEstimate,
    update_density_estimate_single_observation,
    update_density_estimate_multiple_observations,
)
from dmpe.utils.metrics import JSDLoss


def soft_penalty(a: jax.Array, a_max: float = 1.0, penalty_order: int = 2):
    """Computes penalty for the given input. Assumes symmetry in all dimensions.

    Args:
        a (jax.Array): The input array
        a_max (float): The maximum absolute value allowed for the input
        penalty_order (int): The order of the penalty term (default is 2)

    Returns:
        penalty (jax.Array): Resulting penalty value
    """
    penalties = jax.nn.relu(jnp.abs(a) - a_max)
    penalties = penalties**penalty_order

    penalty = jnp.sum(penalties, axis=(-2, -1))
    return jnp.squeeze(penalty)


@eqx.filter_jit
def loss_function(
    model: eqx.Module,
    init_obs: jax.Array,
    init_state: excenvs.CoreEnvironment.State,
    actions: jax.Array,
    density_estimate: DensityEstimate,
    tau: float,
    consider_action_distribution: bool,
    target_distribution: jax.Array,
    penalty_function: Callable,
) -> jax.Array:
    """Predicts a trajectory based on the given actions and the model and computes the
    corresponding loss value based on the Jensen-Shannon divergence loss.

    Args:
        model (eqx.Module): The model to use for the prediction
        init_obs (jax.Array): The initial observation from which to start the simulation
        init_state (excenvs.CoreEnvironment.State): The initial state from which to start
            the simulation (only actually used when using the environment for simulation)
        actions (jax.Array): The actions to apply in each step of the simulation, the length
            of the first dimension of this array determine the length of the output.
        density_estimate (DensityEstimate): The current estimate of the data density
        tau (float): The sampling time for the model
        consider_action_distribution (bool): Whether to consider the action as part of the
            feature space
        target_distribution (jax.Array): The goal distribution of the data. The JSD loss is
            computed w.r.t. this distribution
        penalty_function (Callable): A function that computes a penalty term for the observations
            and actions. This is done to penalize overstepping of the physical constraints of the
            system
    """
    if isinstance(model, eqx.Module):
        observations = simulate_ahead(model=model, init_obs=init_obs, actions=actions, tau=tau)
    else:
        observations, _ = simulate_ahead_with_env(env=model, init_obs=init_obs, init_state=init_state, actions=actions)

    if consider_action_distribution:
        predicted_density_estimate = update_density_estimate_multiple_observations(
            density_estimate, jnp.concatenate([observations[0:-1, :], actions], axis=-1)  # observations
        )
    else:
        predicted_density_estimate = update_density_estimate_multiple_observations(density_estimate, observations)

    loss = JSDLoss(
        p=predicted_density_estimate.p / jnp.sum(predicted_density_estimate.p),
        q=target_distribution / jnp.sum(target_distribution),
    )
    penalty_terms = penalty_function(observations, actions)
    return loss + penalty_terms


@eqx.filter_jit
def optimize_actions(
    loss_function: Callable,
    grad_loss_function: Callable,
    proposed_actions: jax.Array,
    model: eqx.Module,
    optimizer: optax.GradientTransformation | optax.GradientTransformationExtraArgs,
    init_obs: jax.Array,
    init_state: excenvs.CoreEnvironment.State,
    density_estimate: DensityEstimate,
    n_opt_steps: int,
    tau: float,
    consider_action_distribution: bool,
    target_distribution: jax.Array,
    penalty_function: Callable,
):
    """Uses the model to compute the effect of actions onto the observation trajectory to
    optimize the actions w.r.t. the given (gradient of the) loss function.

    Args:
        loss_function (Callable): The loss function to optimize
        grad_loss_function (Callable): The gradient of the loss function w.r.t. the actions
        proposed_actions (jax.Array): The initial proposed actions to optimize
        model (eqx.Module): The model to use for the prediction
        optimizer (optax.GradientTransformation): The optimizer to use for the optimization
        init_obs (jax.Array): The initial observation from which to start the simulation
        init_state (excenvs.CoreEnvironment.State): The initial state from which to start
            the simulation
        density_estimate (DensityEstimate): The momentary estimate of the data density
        n_opt_steps (int): Number of SGD steps per iteration
        tau (float): The sampling time for the model and system
        consider_action_distribution (bool): Whether to consider the action as part of the
            feature space
        target_distribution (jax.Array): The goal distribution of the data. The JSD loss is
            computed w.r.t. this distribution
        penalty_function (Callable): A function that computes a penalty term for the observations
            and actions. This is done to penalize overstepping of the physical constraints of the
            system
    Returns:
        proposed_actions (jax.Array): The optimized actions
        loss (jax.Array): The final loss value
    """
    opt_state = optimizer.init(proposed_actions)

    def body_fun(i, carry):
        proposed_actions, opt_state = carry
        value, grad = grad_loss_function(
            model,
            init_obs,
            init_state,
            proposed_actions,
            density_estimate,
            tau,
            consider_action_distribution,
            target_distribution,
            penalty_function,
        )

        if isinstance(optimizer, optax._src.base.GradientTransformationExtraArgs):

            opt_loss_fn = lambda proposed_actions: loss_function(
                model,
                init_obs,
                init_state,
                proposed_actions,
                density_estimate,
                tau,
                consider_action_distribution,
                target_distribution,
                penalty_function,
            )

            updates, opt_state = optimizer.update(
                grad, opt_state, proposed_actions, grad=grad, value=value, value_fn=opt_loss_fn
            )
        else:
            updates, opt_state = optimizer.update(grad, opt_state, proposed_actions)
        proposed_actions = optax.apply_updates(proposed_actions, updates)

        # proposed_actions = proposed_actions - lr * grad

        return (proposed_actions, opt_state)

    proposed_actions, _ = jax.lax.fori_loop(0, n_opt_steps, body_fun, (proposed_actions, opt_state))

    loss = loss_function(
        model,
        init_obs,
        init_state,
        proposed_actions,
        density_estimate,
        tau,
        consider_action_distribution,
        target_distribution,
        penalty_function,
    )

    return proposed_actions, loss


def optimize_actions_multistart(
    loss_function: Callable,
    grad_loss_function: Callable,
    all_proposed_actions: jax.Array,
    model: eqx.Module,
    optimizer: optax.GradientTransformation | optax.GradientTransformationExtraArgs,
    init_obs: jax.Array,
    init_state: excenvs.CoreEnvironment.State,
    density_estimate: DensityEstimate,
    n_opt_steps: int,
    tau: float,
    consider_action_distribution: bool,
    target_distribution: jax.Array,
    penalty_function: Callable,
):
    """Parallelizes the action optimization function w.r.t. the proposed actions. Thereby,
    enables the use of multistart optimization. Only the result with the best final loss
    value is returned from this function.

    This is done since the gradient based optimization methods employed to solve the
    excitation optimization problem can get stuck in local optima. The multistart aims to
    decrease the probability of this occurring.

    Args:
        loss_function (Callable): The loss function to optimize
        grad_loss_function (Callable): The gradient of the loss function w.r.t. the actions
        all_proposed_actions (jax.Array): The initial proposed actions to optimize. Note that
            compared to the 'optimize_actions' function, 'all_proposed_actions' contains an
            extra dimension for the multistart optimization. Shape: (n_starts, n_sim_steps, input_dim).
        model (eqx.Module): The model to use for the prediction
        optimizer (optax.GradientTransformation): The optimizer to use for the optimization
        init_obs (jax.Array): The initial observation from which to start the simulation
        init_state (excenvs.CoreEnvironment.State): The initial state from which to start
            the simulation
        density_estimate (DensityEstimate): The momentary estimate of the data density
        n_opt_steps (int): Number of SGD steps per iteration
        tau (float): The sampling time for the model and system
        consider_action_distribution (bool): Whether to consider the action as part of the
            feature space
        target_distribution (jax.Array): The goal distribution of the data. The JSD loss is
            computed w.r.t. this distribution
        penalty_function (Callable): A function that computes a penalty term for the observations
            and actions. This is done to penalize overstepping of the physical constraints of the
            system
    Returns:
        proposed_actions (jax.Array): The optimized actions
        loss (jax.Array): The final loss value
    """
    assert all_proposed_actions.ndim == 3, "proposed_actions must have shape (n_starts, n_sim_steps, action_dim)"

    all_optimized_actions, all_losses = jax.vmap(
        optimize_actions,
        in_axes=(None, None, 0, None, None, None, None, None, None, None, None, None, None),
    )(
        loss_function,
        grad_loss_function,
        all_proposed_actions,
        model,
        optimizer,
        init_obs,
        init_state,
        density_estimate,
        n_opt_steps,
        tau,
        consider_action_distribution,
        target_distribution,
        penalty_function,
    )

    best_idx = jnp.argmin(all_losses)
    return all_optimized_actions[best_idx], all_losses[best_idx]


class Exciter(eqx.Module):
    """A class that carries the necessary tools for excitation input computations.

    Args:
        start_optimizing (int): When to start optimizing actions
        loss_function (Callable): The loss function to optimize
        grad_loss_function: The gradient of the loss function w.r.t. the actions as
            a callable function
        excitation_optimizer (optax.GradientTransformation): The optimizer for the
            excitation input computation
        tau (float): The time step length of the simulation
        n_opt_steps (int): Number of SGD steps per iteration
        consider_action_distribution (bool): Whether to consider the action as part of the
            feature vector
        target_distribution (jax.Array): The targeted distribution for the data density
        penalty_function (Callable): A function that computes a penalty term for the observations
            and actions. This is done to penalize overstepping of the physical constraints of the
            system
        clip_action (True): Whether to clip the actions to the action space limits
        n_starts (int): Number of parallel starts for the optimization
        reuse_proposed_actions (True): Whether to reuse the proposed actions from the previous
            iteration as the initial guess for the next step (it is only used for one of the
            n_starts)
    """

    start_optimizing: int
    loss_function: Callable
    grad_loss_function: Callable
    excitation_optimizer: optax.GradientTransformation
    tau: float
    n_opt_steps: int
    consider_action_distribution: bool
    target_distribution: jax.Array
    penalty_function: Callable
    clip_action: bool
    n_starts: int
    reuse_proposed_actions: bool

    @eqx.filter_jit
    def choose_action(
        self,
        obs: jax.Array,
        state: excenvs.CoreEnvironment.State,
        model: eqx.Module,
        density_estimate: DensityEstimate,
        proposed_actions: jax.Array,
        expl_key: jax.random.PRNGKey,
    ) -> tuple[jax.Array, jax.Array, DensityEstimate]:
        """Chooses the next action to take, updates the density estimate and
        proposes future actions.

        Args:
            obs (jax.Array): The momentary observations from which to start
            state (excenvs.CoreEnvironment.State): The momentary state from which to start
            model (eqx.Module): The momentary model of the environment used for the prediction
            density_estimate (DensityEstimate): The current estimate of the data density without
                the current step k
            proposed_actions (jax.Array): An initial proposition of actions to take

        Returns:
            action (jax.Array): The chosen action
            next_proposed_actions (jax.Array): An initial proposition for future actions
            density_estimate (DensityEstimate): The updated density estimate now incorporating
                the current step k
        """

        if self.reuse_proposed_actions:
            n_random_starts = self.n_starts - 1

        expl_key, new_proposed_actions_key, expl_action_key = jax.random.split(expl_key, 3)

        if n_random_starts > 0:
            random_proposed_actions = jax.random.uniform(
                key=new_proposed_actions_key, shape=(n_random_starts, *proposed_actions.shape), minval=-1, maxval=1
            )

            if self.reuse_proposed_actions:
                all_proposed_actions = jnp.concatenate([proposed_actions[None, :], random_proposed_actions], axis=0)
            else:
                all_proposed_actions = random_proposed_actions
        else:
            all_proposed_actions = proposed_actions[None, :]

        proposed_actions, loss = optimize_actions_multistart(
            loss_function=self.loss_function,
            grad_loss_function=self.grad_loss_function,
            all_proposed_actions=all_proposed_actions,
            model=model,
            optimizer=self.excitation_optimizer,
            init_obs=obs,
            init_state=state,
            density_estimate=density_estimate,
            n_opt_steps=self.n_opt_steps,
            tau=self.tau,
            consider_action_distribution=self.consider_action_distribution,
            target_distribution=self.target_distribution,
            penalty_function=self.penalty_function,
        )

        action, next_proposed_actions, density_estimate = self.process_propositions(
            obs=obs,
            density_estimate=density_estimate,
            proposed_actions=proposed_actions,
            expl_action_key=expl_action_key,
        )

        return action, next_proposed_actions, density_estimate, loss, expl_key

    @eqx.filter_jit
    def process_propositions(
        self,
        obs: jax.Array,
        density_estimate: DensityEstimate,
        proposed_actions: jax.Array,
        expl_action_key: jax.random.PRNGKey,
    ) -> tuple[jax.Array, jax.Array, DensityEstimate]:
        """Updates the proposed actions and density estimate for the next iteration after the
        optimization has been performed.

        Args:
            obs (jax.Array): The momentary observation at time step k
            density_estimate (DensityEstimate): The momentary estimate of the data density without
                the momentary step k
            proposed_actions (jax.Array): The (potentially optimized) proposition of actions to take
            expl_action_key (jax.random.PRNGKey): A random key for the exploration action

        Returns:
            action (jax.Array): The chosen action
            next_proposed_actions (jax.Array): An initial proposition for future actions
            density_estimate (DensityEstimate): The updated density estimate now incorporating
                the current step k
        """
        action = proposed_actions[0, :]

        if self.clip_action:
            action = jnp.clip(action, -1, 1)

        next_proposed_actions = proposed_actions.at[:-1, :].set(proposed_actions[1:, :])

        new_proposed_action = jax.random.uniform(key=expl_action_key, minval=-1, maxval=1)
        next_proposed_actions = next_proposed_actions.at[-1, :].set(new_proposed_action)

        # update grid KDE with y_k and u_k
        if self.consider_action_distribution:
            density_estimate = update_density_estimate_single_observation(
                density_estimate, jnp.concatenate([obs, action], axis=-1)
            )
        else:
            density_estimate = update_density_estimate_single_observation(density_estimate, obs)

        return action, next_proposed_actions, density_estimate
