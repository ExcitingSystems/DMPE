import jax
import jax.nn as jnn
import jax.numpy as jnp

import equinox as eqx
import diffrax


class MLP(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        width_size: int,
        depth: int,
        *,
        key: jax.random.PRNGKey,
        **kwargs,
    ):
        """Basic state-space MLP. Prediction is based on observation and action.

        Is generally used here to predict the dx/dt in NODE models.

        Args:
            obs_dim (int): Observation dimensionality
            action_dim (int): Action dimensionality
            width_size (int): Layer width (same for all layer) in the MLP
            depth (int): Number of hidden layers in the MLP
            key (jax.random.PRNGKey): Random key for model initialization
        """
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=(obs_dim + action_dim),
            out_size=obs_dim,
            width_size=width_size,
            depth=depth,
            activation=jnn.leaky_relu,
            key=key,
        )

    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        obs_action = jnp.hstack([obs, action])
        return self.mlp(obs_action)


class NeuralODE(eqx.Module):
    func: MLP
    _solver: diffrax.AbstractSolver

    def __init__(
        self,
        solver: diffrax.AbstractSolver,
        obs_dim: int,
        action_dim: int,
        width_size: int,
        depth: int,
        *,
        key: jax.random.PRNGKey,
        **kwargs,
    ):
        """Neural ordinary differential equation implementation.

        Args:
            solver (diffrax.AbstractSolver): The ODE solver to use for the Neural ODE.
            obs_dim (int): Observation dimensionality
            action_dim (int): Action dimensionality
            width_size (int): Layer width (same for all layer) in the MLP
            depth (int): Number of hidden layers in the MLP
            key (jax.random.PRNGKey): Random key for model initialization
        """

        super().__init__(**kwargs)
        self.func = MLP(obs_dim, action_dim, width_size, depth, key=key)
        self._solver = solver

    def __call__(self, init_obs: jax.Array, actions: jax.Array, tau: float) -> jax.Array:
        """Simulate the observation trajectory using the ODE solver.

        Args:
            init_obs (jax.Array): Initial observation to start the simulation from
            actions (jax.Array): Actions / inputs to be applied
            tau (float): Time step size for the simulation
        """

        args = (actions, None)

        def action_helper(t, args):
            actions = args
            return actions[jnp.array(t / tau, int)]

        def vector_field(t, y, args):
            actions, _ = args

            action = action_helper(t, actions)
            dy_dt = self.func(y, action)
            return tuple(dy_dt)

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = tau * actions.shape[0]

        y0 = tuple(init_obs)
        saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 1 + int(t1 / tau)))
        solution = diffrax.diffeqsolve(term, self._solver, t0, t1, dt0=tau, y0=y0, args=args, saveat=saveat)

        return jnp.transpose(jnp.array(solution.ys))


class NeuralODEPendulum(NeuralODE):
    """Pendulum specific model that deals with the periodic properties of the angle information."""

    def __call__(self, init_obs: jax.Array, actions: jax.Array, tau: float) -> jax.Array:
        observations = super().__call__(init_obs, actions, tau)
        return jnp.stack([(((observations[..., 0] + 1) % 2) - 1), observations[..., 1]], axis=-1)


class NeuralEulerODE(eqx.Module):
    func: MLP

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        width_size: int,
        depth: int,
        *,
        key: jax.random.PRNGKey,
        **kwargs,
    ):
        """Neural ODE model with Euler as the ODE solver.

        Very fast to use compared to the models based on more complex ODE solvers.
        However, will generally be less accurate. In essence this model boils down
        to a difference equation model instead of a true NODE.

        Args:
            solver (diffrax.AbstractSolver): The ODE solver to use for the Neural ODE.
            obs_dim (int): Observation dimensionality
            action_dim (int): Action dimensionality
            width_size (int): Layer width (same for all layer) in the MLP
            depth (int): Number of hidden layers in the MLP
            key (jax.random.PRNGKey): Random key for model initialization
        """
        super().__init__(**kwargs)
        self.func = MLP(obs_dim, action_dim, width_size, depth, key=key)

    def step(self, obs: jax.Array, action: jax.Array, tau: float) -> jax.Array:
        """Takes a single step in the simulation using the Euler method.

        Args:
            obs (jax.Array): Momentary observation
            action (jax.Array): Action to be applied
            tau (float): Time step size for the simulation

        Returns:
            next_obs (jax.Array): The predicted next observation.
        """
        next_obs = obs + tau * self.func(obs, action)
        return next_obs

    def __call__(self, init_obs: jax.Array, actions: jax.Array, tau: float) -> jax.Array:
        """Simulate the observation trajectory using the Euler ODE solver.

        Args:
            init_obs (jax.Array): Initial observation to start the simulation from
            actions (jax.Array): Actions / inputs to be applied
            tau (float): Time step size for the simulation
        """

        def body_fun(carry, action):
            obs = carry
            obs = self.step(obs, action, tau)
            return obs, obs

        _, observations = jax.lax.scan(body_fun, init_obs, actions)
        observations = jnp.concatenate([init_obs[None, :], observations], axis=0)
        return observations


class NeuralEulerODEPendulum(NeuralEulerODE):
    """Pendulum specific model that deals with the periodic properties of the angle information."""

    def step(self, obs: jax.Array, action: jax.Array, tau: float) -> jax.Array:
        next_obs = super().step(obs, action, tau)
        next_obs = jnp.stack([(((next_obs[..., 0] + 1) % 2) - 1), next_obs[..., 1]], axis=-1)
        return next_obs


class NeuralEulerODECartpole(NeuralEulerODE):
    """Cartpole specific model that deals with the periodic properties of the angle information."""

    def step(self, obs: jax.Array, action: jax.Array, tau: float) -> jax.Array:
        next_obs = super().step(obs, action, tau)
        next_obs = jnp.stack(
            [next_obs[..., 0], next_obs[..., 1], (((next_obs[..., 2] + 1) % 2) - 1), next_obs[..., 3]], axis=-1
        )
        return next_obs


class NeuralEulerODEPMSM(NeuralEulerODE):
    """PMSM-specific NODE model that roughly scales the output to fit the normal output range more accurately."""

    def step(self, obs: jax.Array, action: jax.Array, tau: float) -> jax.Array:
        next_obs = obs + tau * self.func(obs, action) * 1e4
        return next_obs
