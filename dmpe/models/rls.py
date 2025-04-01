import jax
import jax.numpy as jnp
import equinox as eqx


class RLS(eqx.Module):
    """RLS estimator based on the description given in [Brosch2021] and [Jakobeit2025]."""

    num_regressors: int
    num_measurements: int
    lambda_: float
    w: jax.Array
    P: jax.Array

    def __init__(self, num_regressors, num_measurements, lambda_, w=None, P=None):
        self.num_regressors = num_regressors
        self.num_measurements = num_measurements
        self.lambda_ = lambda_

        self.w = jnp.zeros((self.num_regressors, self.num_measurements)) if w is None else w
        self.P = jnp.eye(self.num_regressors) if P is None else P

    @staticmethod
    @eqx.filter_jit
    def predict(rls, x):
        """
        Predict the output for a given input using the RLS model.
        """
        y_pred = rls.w.T @ x
        return y_pred

    @staticmethod
    @eqx.filter_jit
    def update(rls, x, d):
        """
        Update function.
        """
        P = rls.P
        w = rls.w

        c = (P @ x) / (rls.lambda_ + jnp.squeeze(x.T @ P @ x))
        w_new = w + c @ (d - w.T @ x).T
        P_new = (jnp.eye(rls.num_regressors) - c @ x.T) @ P / rls.lambda_

        return RLS(rls.num_regressors, rls.num_measurements, rls.lambda_, w=w_new, P=P_new)


class PMSM_RLS(eqx.Module):
    rls: eqx.Module

    def __init__(self, lambda_=0.99, rls=None):
        self.rls = RLS(5, 2, lambda_) if rls is None else rls

    @eqx.filter_jit
    def __call__(self, x):
        return RLS.predict(self.rls, x)

    @classmethod
    @eqx.filter_jit
    def update(cls, pmsm_rls, x, d):
        rls = RLS.update(pmsm_rls.rls, x, d)
        return cls(rls.lambda_, rls)


class SimulationPMSM_RLS(PMSM_RLS):

    @eqx.filter_jit
    def __call__(self, init_obs, actions, tau=None):

        def body_fun(carry, action):
            obs = carry
            rls_in = jnp.concatenate([jnp.squeeze(obs), jnp.squeeze(action), jnp.ones(1)])[..., None]

            obs = super(SimulationPMSM_RLS, self).__call__(rls_in)
            obs = jnp.squeeze(obs)
            return obs, obs

        _, observations = jax.lax.scan(body_fun, jnp.squeeze(init_obs), actions)
        observations = jnp.concatenate([jnp.squeeze(init_obs)[None, :], observations], axis=0)
        return observations

    @classmethod
    @eqx.filter_jit
    def fit(cls, model, k, observations, actions):
        obs = jnp.squeeze(observations[k])
        action = jnp.squeeze(actions[k])
        next_obs = jnp.squeeze(observations[k + 1])

        model_in = jnp.concatenate([obs, action, jnp.ones(1)])[..., None]
        model = cls.update(model, x=model_in, d=next_obs[..., None])
        return model
