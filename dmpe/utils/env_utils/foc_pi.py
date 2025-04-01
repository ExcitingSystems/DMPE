from functools import partial

import jax
import jax.numpy as jnp
from exciting_environments.pmsm.pmsm_env import dq2albet, ROTATION_MAP


def get_advanced_angle(eps, tau_scale, tau, omega):
    return eps + tau_scale * tau * omega


class ClassicController:
    def __init__(self, motor, rpm, saturated, a=4, decoupling=True, tau=1e-4):
        """
        Initializes the ClassicController.

        Args:
            motor (Motor): The motor object containing motor parameters.
            saturated (bool): Indicates whether the motor is saturated.
            a (int, optional): A parameter for the controller. Defaults to 4.
            decoupling (bool, optional): Indicates whether decoupling is enabled. Defaults to True.
            tau (float, optional): The time constant for the controller. Defaults to 1e-4.

        Attributes:
            p_gain (float): Proportional gain for the controller.
            i_gain (float): Integral gain for the controller.
            tau (float): The time constant for the controller.
            a (int): A parameter for the controller.
            motor (Motor): The motor object containing motor parameters.
            L_dq (float): Inductance in the dq-axis (if not saturated).
            psi_dq (float): Flux linkage in the dq-axis (if not saturated).
            batch_size (int): The batch size for processing.
            decoupling (bool): Indicates whether decoupling is enabled.
            saturated (bool): Indicates whether the motor is saturated.
            interpolators (Interpolator): Interpolators for the motor (if saturated).
            u_s_0 (float): Initial control signal.
            integrated (float): Integrated error for the controller.
        """
        self.tau: float = tau
        self.a: int = a

        self.motor = motor
        self.rpm: float = rpm

        if not saturated:
            self.L_dq: float = motor.l_dq
            self.psi_dq: float = motor.psi_dq

        self.batch_size: int = None
        self.decoupling: bool = decoupling
        self.saturated: bool = saturated
        if self.saturated:
            self.interpolators = motor.LUT_interpolators
        self.u_s_0 = 0

        self.tune()

    def tune(self):
        """
        Tunes the controller by setting the proportional and integral gains.

        If the motor is not saturated, the proportional gain (p_gain) and integral gain (i_gain)
        are calculated based on the motor's inductance (L_dq), the parameter 'a', and the time constant (tau).

        Returns:
            None
        """
        if not self.saturated:
            self.p_gain = 1 / (self.a * 1.5 * self.tau) * self.L_dq
            self.i_gain = self.p_gain / ((self.a) ** 2 * 1.5 * self.tau)

    def check_constraints(self, e, eps, integrated, p_gain, i_gain, u_s_0):
        """
        Checks whether operation points are allowed to be integrated and returns a mask.

        The mask indicates whether the operation points are within the allowed constraints.
        A value of 0 indicates "nok" and 1 indicates "ok".

        Args:
            e (np.ndarray): The error array.
            eps (float): The epsilon value.

        Returns:
            np.ndarray: A mask array where 0s indicate not okay and 1s indicate okay.
        """

        mask = jnp.zeros_like(e)
        u_dq = p_gain * e + i_gain * (integrated + e * self.tau) + u_s_0

        u_dq_norm = self.motor.env_properties.action_normalizations.u_d.normalize(u_dq)

        u_albet_norm = dq2albet(
            u_dq_norm,
            get_advanced_angle(eps, 0.5, self.tau, (3 * self.rpm / 60 * 2 * jnp.pi)),
        )
        u_albet_c = u_albet_norm[:, 0] + 1j * u_albet_norm[:, 1]
        idx = (jnp.sin(jnp.angle(u_albet_c)[..., jnp.newaxis] - 2 / 3 * jnp.pi * jnp.arange(3)) >= 0).astype(int)
        rot_vecs = ROTATION_MAP[idx[:, 0], idx[:, 1], idx[:, 2]]
        u_albet_c = jnp.multiply(u_albet_c, rot_vecs)
        cond1 = (jnp.abs(u_albet_c.real) < 2 / 3) & (
            (u_albet_c.imag > 0) & (u_albet_c.imag < 2 / 3 * jnp.sqrt(3))
        )  # Inside
        cond2 = (
            ~(jnp.abs(u_albet_c.real) < 2 / 3) | ((u_albet_c.imag < 0) | (u_albet_c.imag > 2 / 3 * jnp.sqrt(3)))
        ) & jnp.all(
            e < 0, axis=1
        )  # outside but neg

        mask = jnp.logical_or(cond1, cond2)
        return mask

    def integrate(self, integrated, e, mask):
        """
        Integrates the error over time with the given mask.

        Args:
            e (np.ndarray): The error array.
            mask (np.ndarray): The mask array indicating which errors to integrate.

        Returns:
            None
        """
        # integrated += e * self.tau * mask
        integrated = (integrated + e * self.tau) * mask
        return integrated

    @partial(jax.jit, static_argnums=[0])
    def __call__(self, obs, integrated):
        """
        Computes the control action based on the observed state.

        Args:
            obs (np.ndarray): The observation array containing the normalized state, epsilon, and reference values.

        Returns:
            np.ndarray: The normalized control action voltages.
        """
        # Extract normalized state, epsilon, and reference values from the observation
        state_norm = obs[:, :2]
        eps = obs[:, 2:3]
        reference_norm = obs[:, 3:5]

        # Calculate the error between the reference and the current state
        state = jnp.concatenate(
            [
                self.motor.env_properties.physical_normalizations.i_d.denormalize(state_norm[:, 0]),
                self.motor.env_properties.physical_normalizations.i_q.denormalize(state_norm[:, 1]),
            ],
            axis=0,
        )[None]
        reference = jnp.concatenate(
            [
                self.motor.env_properties.physical_normalizations.i_d.denormalize(reference_norm[:, 0]),
                self.motor.env_properties.physical_normalizations.i_q.denormalize(reference_norm[:, 1]),
            ],
            axis=0,
        )[None]

        e = reference - state

        if self.saturated:
            # Tune gains if the motor is saturated
            p_d = {q: interp(state) for q, interp in self.interpolators.items()}
            # Neglect non-main-diagonal entries of L_diff
            L_dq = jnp.column_stack([p_d[q] for q in ["L_dd", "L_qq"]])
            psi_dq = jnp.column_stack([p_d[q] for q in ["Psi_d", "Psi_q"]])
            p_gain = 1 / (self.a * 1.5 * self.tau) * L_dq
            p_gain_help = 1 / (self.a * 1.5 * self.tau) * L_dq
            i_gain = 0.3 * p_gain_help / ((self.a) ** 2 * 1.5 * self.tau)

        if self.decoupling:
            # Apply decoupling if enabled
            i_dq = state
            q = jnp.array([0, -1, 1, 0]).reshape(2, 2)
            if self.saturated:
                # Calculate the initial control signal for saturated motor
                u_s_0 = (3 * self.rpm / 60 * 2 * jnp.pi) * jnp.einsum("ij,bj->bi", q, psi_dq)
            else:
                # Calculate the initial control signal for non-saturated motor
                u_s_0 = (3 * self.rpm / 60 * 2 * jnp.pi) * jnp.einsum("ij,bj->bi", q, i_dq * L_dq + psi_dq)
        else:
            u_s_0 = self.u_s_0

        # Check constraints and get the integration allowance mask
        integration_allowance_mask = self.check_constraints(e, eps, integrated, p_gain, i_gain, u_s_0)
        # Integrate the error with the mask
        integrated = self.integrate(integrated, e, integration_allowance_mask)
        # Calculate the control action
        u_dq = p_gain * e + i_gain * integrated + u_s_0

        # Normalize the control action voltages
        return self.motor.env_properties.action_normalizations.u_d.normalize(u_dq), integrated

    def reset(self, batch_size):
        """
        Resets the controller's state for a new batch of data.

        Args:
            batch_size (int): The size of the new batch.

        Returns:
            None
        """
        self.batch_size = batch_size

        return jnp.zeros((self.batch_size, 2))

    def parameters(self):
        """
        Returns the parameters of the controller.

        Returns:
            list: An empty list of parameters.
        """
        return []
