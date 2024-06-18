import jax
import jax.nn as jnn
import jax.numpy as jnp

import equinox as eqx
import diffrax


class MLP(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, obs_dim, action_dim, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=(obs_dim + action_dim),
            out_size=obs_dim,
            width_size=width_size,
            depth=depth,
            activation=jnn.leaky_relu,
            key=key,
        )

    def __call__(self, obs, action):
        obs_action = jnp.hstack([obs, action])
        return self.mlp(obs_action)


class NeuralODE(eqx.Module):
    func: MLP
    _solver: diffrax.AbstractSolver

    def __init__(self, solver, obs_dim, action_dim, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = MLP(obs_dim, action_dim, width_size, depth, key=key)
        self._solver = solver

    def __call__(self, init_obs, actions, tau):

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
    def __call__(self, init_obs, actions, tau):
        observations = super().__call__(init_obs, actions, tau)
        return jnp.stack([(((observations[..., 0] + 1) % 2) - 1), observations[..., 1]], axis=-1)


class NeuralEulerODE(eqx.Module):
    func: MLP

    def __init__(self, obs_dim, action_dim, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = MLP(obs_dim, action_dim, width_size, depth, key=key)

    def step(self, obs, action, tau):
        next_obs = obs + tau * self.func(obs, action)
        return next_obs

    def __call__(self, init_obs, actions, tau):

        def body_fun(carry, action):
            obs = carry
            obs = self.step(obs, action, tau)
            return obs, obs

        _, observations = jax.lax.scan(body_fun, init_obs, actions)
        observations = jnp.concatenate([init_obs[None, :], observations], axis=0)
        return observations


class NeuralEulerODEPendulum(NeuralEulerODE):
    """Pendulum specific model that deals with the periodic properties of the angle information."""

    def step(self, obs, action, tau):
        next_obs = super().step(obs, action, tau)
        next_obs = jnp.stack([(((next_obs[..., 0] + 1) % 2) - 1), next_obs[..., 1]], axis=-1)
        return next_obs
