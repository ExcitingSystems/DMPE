import abc
from functools import partial

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import equinox as eqx
from dmpe.evaluation.utils import valid_space_grid
from dmpe.models.models import NeuralEulerODE
from dmpe.models.model_utils import simulate_ahead_with_env
import exciting_environments as excenvs


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
        observations, _ = simulate_ahead_with_env(self.model, init_obs, init_state, actions)
        return eqx.filter_vmap(self.featurize)(observations)


class PredictionComparison(eqx.Module):
    grid: jax.Array
    action_dim: float
    obs_dim: float
    tau: float

    def __init__(self, grid, action_dim, obs_dim, tau):
        assert grid.shape[-1] == (action_dim + obs_dim), (
            "The grid dimension does not fit the action_dim and obs_dim."
            + f"Grid dimension should be action_dim+obs_dim, but {grid.shape[-1]} and {action_dim}+{obs_dim} are given."
        )
        self.grid = grid
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.tau = tau

    @eqx.filter_jit
    def __call__(self, model, model_gt):

        observations = self.grid[:, : self.obs_dim]
        actions = self.grid[:, self.obs_dim :]

        pred = eqx.filter_vmap(model.step, in_axes=(0, 0, None))(observations, actions, self.tau)
        pred_gt = eqx.filter_vmap(model_gt.step, in_axes=(0, 0, None))(observations, actions, self.tau)

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

    @eqx.filter_jit
    def __call__(self, model, model_gt):
        pred = jax.vmap(model.gradient, in_axes=(0, 0))(self.grid[:, : self.obs_dim], self.grid[:, self.obs_dim :])
        pred_gt = jax.vmap(model_gt.gradient, in_axes=(0, 0))(
            self.grid[:, : self.obs_dim], self.grid[:, self.obs_dim :]
        )

        return (pred - pred_gt), jnp.mean((pred - pred_gt) ** 2)


class RolloutComparison(eqx.Module):
    control_law: callable
    penalty_function: callable
    tau: float
    env: excenvs.CoreEnvironment
    sequence_length: int

    def _generate_actions(
        self,
        init_obs: jax.Array,
        env: excenvs.CoreEnvironment,
        control_law: callable,
        penalty_function: callable,
        sequence_length: int,
        key: jax.random.PRNGKey,
    ):

        action_dim = env.action_dim
        obs_dim = env.reset(env.env_properties)[0].shape[0]

        init_state = env.generate_state_from_observation(init_obs, env.env_properties)
        observations = jnp.zeros((sequence_length, obs_dim))
        actions = jnp.zeros((sequence_length, action_dim))

        key, action_key = jax.random.split(key, 2)
        last_action = jax.random.normal(action_key, shape=(action_dim,))

        def body_fun(i, carry):
            last_action, state, observations, actions, key = carry
            action, key = control_law(env, penalty_function, last_action, state, key)
            obs, state = env.step(state, action, env.env_properties)

            observations = observations.at[i].set(obs)
            actions = actions.at[i].set(action)
            return action, state, observations, actions, key

        _, _, observations, actions, _ = jax.lax.fori_loop(
            0, sequence_length, body_fun, (last_action, init_state, observations, actions, key)
        )
        observations = jnp.concatenate([init_obs[None], observations], axis=0)

        return observations, actions

    @eqx.filter_jit
    def __call__(
        self,
        init_obs: jax.Array,
        model: ModelWrapper,
        model_gt: ModelWrapper,
        key: jax.random.PRNGKey,
    ):
        key, action_key = jax.random.split(key, 2)
        action_keys = jax.random.split(action_key, init_obs.shape[0])

        env_observations, actions = eqx.filter_vmap(self._generate_actions, in_axes=(0, None, None, None, None, 0))(
            init_obs,
            self.env,
            self.control_law,
            self.penalty_function,
            self.sequence_length,
            action_keys,
        )
        pred_gt = eqx.filter_vmap(model_gt.rollout, in_axes=(0, 0, None))(init_obs, actions, self.tau)
        pred = eqx.filter_vmap(model.rollout, in_axes=(0, 0, None))(init_obs, actions, self.tau)

        return (env_observations, pred_gt, pred, key), jnp.mean((pred - pred_gt) ** 2)


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


def visualize_model_prediction_performance(wrapped_model, model_evaluator: ModelEvaluator, labels: list[str]):
    difference_map, _ = model_evaluator.default_metrics["pred_comp"](
        wrapped_model,
        model_evaluator.gt_model,
    )

    n_features = model_evaluator.obs_dim + model_evaluator.act_dim
    reshaped_difference_map = difference_map.reshape([model_evaluator.validation_points_per_dim] * n_features + [-1])
    # abs_map = jnp.mean(jnp.abs(reshaped_difference_map) ** 2, axis=-1)
    abs_map = jnp.linalg.norm(reshaped_difference_map, axis=-1)

    fig, axs = plt.subplots(nrows=n_features, ncols=n_features, figsize=(6, 6), sharex=True, sharey=True)

    feature_indices = jnp.arange(0, n_features, 1).tolist()

    for i in range(n_features):
        for j in range(n_features):

            axs[j, i].grid(True)
            axs[j, i].set_xlim(-1.1, 1.1)
            axs[j, i].set_ylim(-1.1, 1.1)

            reduction_indices = [f_idx for f_idx in feature_indices if not (f_idx == i or f_idx == j)]
            if len(reduction_indices) == n_features - 1:
                continue

            image = jnp.mean(jnp.abs(abs_map), axis=tuple(reduction_indices))

            if i < j:
                image = jnp.transpose(image)

            axs[j, i].imshow(image, origin="lower", extent=[-1, 1, -1, 1])
            axs[j, 0].set_ylabel(labels[j])

        axs[-1, i].set_xlabel(labels[i])
    fig.tight_layout()

    return fig, axs
