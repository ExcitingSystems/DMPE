from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import jax_dataclasses as jdc
from dmpe.evaluation.utils import valid_space_grid
import abc


class PredictionComparison(eqx.Module):
    grid: jax.Array
    action_dim: float
    obs_dim: float
    tau: float

    def __init__(self, grid, action_dim, obs_dim, tau):
        assert grid.shape[-1] == (
            action_dim + obs_dim
        ), f"The grid dimension does not fit the action_dim and obs_dim. Grid dimension should be action_dim+obs_dim, but {grid.shape[-1]} and {action_dim}+{obs_dim} are given"
        self.grid = grid
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.tau = tau

    def __call__(self, model, model_gt):
        pred = jax.vmap(model.step, in_axes=(0, 0, None))(
            self.grid[:, : self.obs_dim], self.grid[:, self.obs_dim :], self.tau
        )
        pred_gt = jax.vmap(model_gt.step, in_axes=(0, 0, None))(
            self.grid[:, : self.obs_dim], self.grid[:, self.obs_dim :], self.tau
        )

        return (pred - pred_gt), jnp.mean((pred - pred_gt) ** 2)


class GradientComparison(eqx.Module):
    grid: jax.Array
    action_dim: float
    obs_dim: float

    def __init__(self, grid, action_dim, obs_dim):
        assert grid.shape[-1] == (
            action_dim + obs_dim
        ), f"The grid dimension does not fit the action_dim and obs_dim. Grid dimension should be action_dim+obs_dim, but {grid.shape[-1]} and {action_dim}+{obs_dim} are given"
        self.grid = grid
        self.action_dim = action_dim
        self.obs_dim = obs_dim

    def __call__(self, model, model_gt):
        pred = jax.vmap(model.gradient, in_axes=(0, 0))(self.grid[:, : self.obs_dim], self.grid[:, self.obs_dim :])
        pred_gt = jax.vmap(model_gt.gradient, in_axes=(0, 0))(
            self.grid[:, : self.obs_dim], self.grid[:, self.obs_dim :]
        )

        return (pred - pred_gt), jnp.mean((pred - pred_gt) ** 2)


class ModelEvaluator:
    def __init__(self, constraint_function, gt_model, obs_dim, act_dim, validation_points_per_dim, tau):
        self.constraint_function = constraint_function
        self.gt_model = gt_model
        self.constraint_data_space_grid = valid_space_grid(
            constraint_function, obs_dim + act_dim, validation_points_per_dim, -1, 1
        )
        # create default metrics with default params
        self.default_metrics = {
            "pred_comp": PredictionComparison(
                grid=self.constraint_data_space_grid,
                action_dim=act_dim,
                obs_dim=obs_dim,
                tau=tau,
            ),
            "gradient_comp": GradientComparison(
                grid=self.constraint_data_space_grid,
                action_dim=act_dim,
                obs_dim=obs_dim,
            ),
        }


class ModelWrapper(abc.ABC):
    """
    A base class for wrapping models, providing an interface for step-wise predictions,
    gradient computations, and simulation rollouts.
    """

    model: eqx.Module

    def __init__(self, model, **kwargs):
        self.model = model

    @abc.abstractmethod
    def step(self, obs, action, tau):
        """
        Perform a one-step prediction given the current observation, action, and time step.

        Args:
            obs: The current state or observation.
            action: The action taken at the current state.
            tau: The time step for the prediction.

        Returns:
            The predicted next observation after applying the action.
        """
        return

    @abc.abstractmethod
    def gradient(self, obs, action):
        """
        Compute the gradient of the state with respect to time.

        Args:
            obs: The current state or observation x(t).
            action: The action u(t) applied at the current state.

        Returns:
            The gradient dx(t)/dt = f(x(t), u(t)).
        """
        return

    @abc.abstractmethod
    def rollout(self, init_obs, actions, tau):
        """
        Simulate a trajectory starting from an initial observation and a sequence of actions.

        Args:
            init_obs: The initial state or observation.
            actions: A sequence of actions to apply over time.
            tau: The time step for each action in the trajectory.

        Returns:
            A sequence of observations representing the simulation rollout.
        """
        return
