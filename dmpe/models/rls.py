"""RLS estimator based on the description given in, e.g., ["Data-Driven Recursive Least Squares Estimation
for Model Predictive Current Control of Permanent Magnet Synchronous Motors", A. Brosch et al., 2021] and
["Universal Direct Torque Controller for Permanent Magnet Synchronous Motors via Meta-Reinforcement
Learning", D. Jakobeit et al., 2025].
"""

import jax
import jax.numpy as jnp
import equinox as eqx


class RLS(eqx.Module):
    num_regressors: int
    num_measurements: int
    lambda_: float
    w: jax.Array
    P: jax.Array

    def __init__(
        self,
        num_regressors: int,
        num_measurements: int,
        lambda_: float,
        w: jax.Array | None = None,
        P: jax.Array | None = None,
    ):
        """Creates a basic RLS model. Either from existing w and P arrays or by initializing
        them to zero and identity matrix, if they are not specified.

        Args:
            num_regressors (int): Number of regressors, i.e., number of inputs
            num_measurements (int): Number of measurements, i.e., number of outputs
            lambda_ (float): Forgetting factor, that determines how quickly the parameters
                are overwritten based on new regressors and measurements
            w (jax.Array | None): Weight matrix with shape (num_regressors, num_measurements)
                or None, which defaults to a corresponding zero matrix
            P (jax.Array | None): Covariance matrix with shape (num_regressors, num_regressors)
                or None, which defaults to a corresponding identity matrix
        """
        self.num_regressors = num_regressors
        self.num_measurements = num_measurements
        self.lambda_ = lambda_

        self.w = jnp.zeros((self.num_regressors, self.num_measurements)) if w is None else w
        self.P = jnp.eye(self.num_regressors) if P is None else P

    @eqx.filter_jit
    def predict(self, x: jax.Array) -> jax.Array:
        """Predict the output for a given regressor vector using the RLS model. Uses the parameter
        matrix currently stored in the RLS model.

        Args:
            x (jax.Array): The regressor vector with shape (num_regressors, 1) to use for the prediction

        Returns:
            y_pred (jax.Array): The predicted output vector with shape (num_measurements, 1)
        """
        y_pred = self.w.T @ x
        return y_pred

    @staticmethod
    @eqx.filter_jit
    def update(rls: "RLS", x: jax.Array, d: jax.Array) -> "RLS":
        """Update the parameters of the RLS model.

        NOTE that this is done in a static method that takes the momentary RLS model
        as an input and creates a new instance with new parameters. This is done since
        eqx.Modules are immutable.

        Args:
            rls (RLS): The momentary RLS model to update
            x (jax.Array): The regressor vector with shape (num_regressors, 1)
            d (jax.Array): The measured output value with shape (num_measurements, 1)

        Returns:
            rls (RLS): The updated RLS model
        """
        P = rls.P
        w = rls.w

        c = (P @ x) / (rls.lambda_ + jnp.squeeze(x.T @ P @ x))
        w_new = w + c @ (d - w.T @ x).T
        P_new = (jnp.eye(rls.num_regressors) - c @ x.T) @ P / rls.lambda_

        return RLS(rls.num_regressors, rls.num_measurements, rls.lambda_, w=w_new, P=P_new)


class PMSM_RLS(eqx.Module):
    rls: eqx.Module

    def __init__(self, lambda_: float = 0.99, rls: RLS | None = None):
        """RLS model specifically for the PMSM.

        The num_regressors is set to 5 with the expected regressor vector [i_d, i_q, u_d, u_q, 1].
        The num_measurements is set to 2 with the expected measurement vector [i_d, i_q].

        Args:
            lambda_ (float): Forgetting factor, that determines how quickly the parameters
                are overwritten based on new inputs
            rls (RLS): An existing RLS model or None if a new one should be created
        """
        self.rls = RLS(5, 2, lambda_) if rls is None else rls

    @eqx.filter_jit
    def __call__(self, x: jax.Array) -> jax.Array:
        """Use the underlying RLS model to perform a prediction.

        Args:
            x (jax.Array): The regressor vector with shape (num_regressors, 1)
        """
        return self.rls.predict(x)

    @classmethod
    @eqx.filter_jit
    def update(cls, pmsm_rls: "PMSM_RLS", x: jax.Array, d: jax.Array) -> "PMSM_RLS":
        """Update the PMSM specific RLS model.

        Args:
            pmsm_rls (PMSM_RLS): The momentary PMSM-RLS model
            x (jax.Array): The regressor vector with shape (num_regressors, 1)
            d (jax.Array): The measured output value with shape (num_measurements, 1)

        Returns:
            new_pmsm_rls (PMSM_RLS): The updated PMSM-RLS model
        """

        rls = RLS.update(pmsm_rls.rls, x, d)
        return cls(rls.lambda_, rls)

    @property
    def num_regressors(self) -> int:
        return self.rls.num_regressors

    @property
    def num_measurements(self) -> int:
        return self.rls.num_measurements


class SimulationPMSM_RLS(PMSM_RLS):
    """Wraps the PMSM_RLS model to perform multistep simulation.

    This API is needed to make the RLS model properly usable in DMPE.
    """

    @eqx.filter_jit
    def __call__(self, init_obs: jax.Array, actions: jax.Array, tau=None) -> jax.Array:
        """Simulate multiple steps ahead by applying actions starting from the initial observation.

        Args:
            init_obs (jax.Array): The initial observation, expected to be [i_d, i_q] with shape (2,)
                (will be squeezed in the function)
            actions (jax.Array): The actions to be applied, expected to be a sequence of [u_d, u_q]
                value-pairs. The shape is expected to be (n_sim_steps, 2). The function iterates
                along the first dimension and squeezes the resulting values.
            tau: NOTE This value is not actually used in this version and is only added since it is
                part of the model API for DMPE.

        Returns:
            observations (jax.Array): The simulation / prediction result with shape (n_sim_steps, 2)
        """

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
    def fit(
        cls,
        model: "SimulationPMSM_RLS",
        k: jax.Array,
        observations: jax.Array,
        actions: jax.Array,
    ) -> "SimulationPMSM_RLS":
        """Update the RLS model with the observations and actions at index k (generally this will be
        the most recent interaction with the system).

        Args:
            model (SimulationPMSM_RLS): The momentary SimulationPMSM_RLS model to update
            k (jax.Array): The index of the observations and actions to use for the update
            observations (jax.Array): The observations from the environment, expected to be a sequence
                of [i_d, i_q] value-pairs with shape (N, 2) with N > k
            actions (jax.Array): The actions taken in the environment, expected to be a sequence of
                [u_d, u_q] value-pairs. The shape is expected to be (N-1, 2) with N > k

        Returns:
            model (SimulationPMSM_RLS): The updated SimulationPMSM_RLS model
        """
        obs = jnp.squeeze(observations[k])
        action = jnp.squeeze(actions[k])
        next_obs = jnp.squeeze(observations[k + 1])

        model_in = jnp.concatenate([obs, action, jnp.ones(1)])[..., None]
        model = cls.update(model, x=model_in, d=next_obs[..., None])
        return model
