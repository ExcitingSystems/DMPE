import jax.nn as jnn
import jax.numpy as jnp

import equinox as eqx


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


class NeuralEulerODE(eqx.Module):
    func: MLP

    def __init__(self, obs_dim, action_dim, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = MLP(obs_dim, action_dim, width_size, depth, key=key)

    def __call__(self, obs, action, tau):
        next_obs = obs + tau * self.func(obs, action)
        return next_obs


class NeuralEulerODEPendulum(NeuralEulerODE):

    def __call__(self, obs, action, tau):
        next_obs = super().__call__(obs, action, tau)
        next_obs = jnp.stack(
            [(((next_obs[..., 0] + 1) % 2) - 1), next_obs[..., 1]],
            axis=-1
        )
        return next_obs
