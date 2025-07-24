import abc

import jax
import jax.numpy as jnp
import equinox as eqx
from dmpe.evaluation.utils import valid_space_grid
from dmpe.models.models import NeuralEulerODE
import exciting_environments as excenvs


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
        self.validation_points_per_dim = validation_points_per_dim
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.tau = tau

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
    featurize: callable

    def __init__(self, model, featurize, **kwargs):
        self.model = model
        self.featurize = featurize

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


class NodeModelWrapper(ModelWrapper):
    """Wraps a DMPE NODE model for comparison."""

    model: NeuralEulerODE
    featurize: callable

    def step(self, obs, action, tau):
        return self.featurize(self.model.step(obs, action, tau))

    def gradient(self, obs, action):
        return self.model.func(obs, action)

    def rollout(self, init_obs, actions, tau):
        pred = self.model(init_obs, actions, tau)
        return eqx.filter_vmap(self.featurize)(pred)


class EnvWrapper(ModelWrapper):
    """Wraps an exciting_environments env for comparison.

    can you do this by differentiating the step function by time?
    """

    model: excenvs.CoreEnvironment
    featurize: callable

    @eqx.filter_jit
    def step(self, obs, action, tau):
        assert tau == self.model.tau
        state = self.model.generate_state_from_observation(obs, self.model.env_properties)
        next_obs, _ = self.model.step(state, action, self.model.env_properties)
        return self.featurize(next_obs)

    @eqx.filter_jit
    def gradient(self, obs, action):
        raise NotImplementedError

    @eqx.filter_jit
    def rollout(self, init_obs, actions, tau):
        init_state = self.model.generate_state_from_observation(init_obs, self.model.env_properties)
        observations, _, _ = self.model.sim_ahead(init_state, actions, self.model.env_properties, tau, tau)
        return eqx.filter_vmap(self.featurize)(observations)
