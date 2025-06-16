"""Smaller utils specifically for the experiments with the pmsm environment."""

import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp

from exciting_environments.pmsm.pmsm_env import PMSM
from dmpe.excitation.excitation_utils import soft_penalty
from dmpe.evaluation.plotting_utils import plot_sequence


class ExcitingPMSM(PMSM):

    def __init__(self, initial_rpm, *args, **kwargs):
        self.initial_rpm = initial_rpm
        super().__init__(*args, **kwargs)

    def generate_observation(self, system_state, env_properties):
        physical_normalizations = env_properties.physical_normalizations

        obs = jnp.hstack(
            (
                physical_normalizations.i_d.normalize(system_state.physical_state.i_d),
                physical_normalizations.i_q.normalize(system_state.physical_state.i_q),
            )
        )
        return obs

    def init_state(self, env_properties, rng=None, vmap_helper=None):
        """Returns default initial state for all batches."""
        phys = self.PhysicalState(
            u_d_buffer=0.0,
            u_q_buffer=0.0,
            epsilon=0.0,
            i_d=-env_properties.physical_normalizations.i_d.max / 2,
            i_q=0.0,
            torque=0.0,
            omega_el=2 * jnp.pi * 3 * self.initial_rpm / 60,
        )
        subkey = jnp.nan
        additions = None
        ref = self.PhysicalState(
            u_d_buffer=jnp.nan,
            u_q_buffer=jnp.nan,
            epsilon=jnp.nan,
            i_d=jnp.nan,
            i_q=jnp.nan,
            torque=jnp.nan,
            omega_el=jnp.nan,
        )
        return self.State(physical_state=phys, PRNGKey=subkey, additions=additions, reference=ref)


def PMSM_penalty(env, observations, actions, penalty_order=2):

    # action_penalty = soft_penalty(actions, a_max=0.8660254, penalty_order=penalty_order)
    # action_penalty = soft_penalty(actions, a_max=1, penalty_order=penalty_order)

    if actions is None:
        action_penalty = 0
    else:
        physical_u_d = env.env_properties.action_normalizations.u_d.denormalize(actions[..., 0]) / 400
        physical_u_q = env.env_properties.action_normalizations.u_q.denormalize(actions[..., 1]) / 400

        action_penalty = jax.nn.relu(jnp.sqrt(physical_u_d**2 + physical_u_q**2) - 1 / jnp.sqrt(3)) ** penalty_order
        action_penalty = jnp.sum(action_penalty)

    physical_i_d = env.env_properties.physical_normalizations.i_d.denormalize(observations[..., 0])
    physical_i_q = env.env_properties.physical_normalizations.i_q.denormalize(observations[..., 1])

    a = physical_i_d / jnp.abs(env.env_properties.physical_normalizations.i_d.min)
    b = physical_i_q / jnp.abs(env.env_properties.physical_normalizations.i_q.max)

    obs_penalty = jax.nn.relu(jnp.sqrt(a**2 + b**2) - 1) ** penalty_order
    obs_penalty = jnp.sum(obs_penalty)
    i_d_penalty = jnp.sum(jax.nn.relu(a)) ** penalty_order

    return (obs_penalty + i_d_penalty + action_penalty) * 1e3
    # return (obs_penalty + action_penalty) * 1e3


def plot_current_constraints(fig, ax, i_d_normalizer, i_q_normalizer):
    i_d_plot = np.linspace(i_d_normalizer.min, i_d_normalizer.max, 1000)
    i_q_plot = np.sqrt(i_q_normalizer.max**2 - i_d_plot**2)

    ax.plot(i_d_plot, i_q_plot, "k")
    ax.plot(i_d_plot, -i_q_plot, "k")
    ax.plot(np.zeros(2), np.array([i_q_normalizer.min, i_q_normalizer.max]), "k")
    return fig, ax


def plot_sequence_with_constraints(env, observations, actions):
    i_d_normalizer = env.env_properties.physical_normalizations.i_d
    i_q_normalizer = env.env_properties.physical_normalizations.i_q

    physical_i_d = i_d_normalizer.denormalize(observations[..., 0])
    physical_i_q = i_q_normalizer.denormalize(observations[..., 1])

    print("max i_q current:", env.env_properties.physical_normalizations.i_q.max, "A")
    print("min i_q current:", env.env_properties.physical_normalizations.i_q.min, "A")

    fig, axs = plot_sequence(
        observations=jnp.vstack([physical_i_d, physical_i_q]).T,
        actions=jnp.vstack(
            [
                env.env_properties.action_normalizations.u_d.denormalize(actions[..., 0]),
                env.env_properties.action_normalizations.u_q.denormalize(actions[..., 1]),
            ]
        ).T,
        tau=env.tau,
        obs_labels=["i_d", "i_q"],
        action_labels=["u_d", "u_q"],
    )
    t = jnp.linspace(0, observations.shape[0] - 1, observations.shape[0]) * env.tau
    axs[0].plot(t, env.env_properties.physical_normalizations.i_d.denormalize(np.ones(observations.shape[0])), "k")
    axs[0].plot(t, env.env_properties.physical_normalizations.i_d.denormalize(-np.ones(observations.shape[0])), "k")
    axs[0].plot(t, env.env_properties.physical_normalizations.i_q.denormalize(np.ones(observations.shape[0])), "k")
    axs[0].set_ylim(-350, 350)

    axs[1].set_xlim(-350, 100)
    axs[1].set_ylim(-350, 350)
    axs[2].plot(t[:-1], env.env_properties.action_normalizations.u_d.denormalize(np.ones(actions.shape[0])), "k")
    axs[2].plot(t[:-1], env.env_properties.action_normalizations.u_d.denormalize(-np.ones(actions.shape[0])), "k")

    plot_current_constraints(fig, axs[1], i_d_normalizer, i_q_normalizer)

    return fig, axs


def plot_constraints_induced_voltage(env, physical_i_d, physical_i_q, w_el, saturated=True, show_torque=False):
    """Plot the constraints onto the currents due to the induced voltage."""
    p = env.env_properties.static_params.p
    r_s = env.env_properties.static_params.r_s
    psi_p = env.env_properties.static_params.psi_p

    if saturated:
        Psi_d = env.pmsm_lut["Psi_d"][1:-1, 1:-1]
        Psi_q = env.pmsm_lut["Psi_q"][1:-1, 1:-1]
        i_d_vec = np.squeeze(env.pmsm_lut["i_d_vec"])
        i_q_vec = np.squeeze(env.pmsm_lut["i_q_vec"])

    if not saturated:
        # initialize flux matrices and current vectors or something

        l_d = env.env_properties.static_params.l_d
        l_q = env.env_properties.static_params.l_q

    udc = 400  # This should really be part of the properties somehow

    U = np.zeros_like(Psi_d)
    I = np.zeros_like(Psi_d)
    T = np.zeros_like(Psi_d)
    Id = np.zeros_like(Psi_d)
    Iq = np.zeros_like(Psi_d)

    # Loop over current vectors
    w = w_el
    for dd in range(len(i_d_vec)):
        for qq in range(len(i_q_vec)):
            id_val = float(i_d_vec[dd])
            iq_val = float(i_q_vec[qq])

            if saturated:
                psid = Psi_d[qq, dd]
                psiq = Psi_q[qq, dd]
            else:
                psid = psi_p + l_d * id_val
                psiq = l_q * iq_val

            ud = r_s * id_val - w * psiq
            uq = r_s * iq_val + w * psid

            Id[qq, dd] = id_val
            Iq[qq, dd] = iq_val
            U[qq, dd] = np.sqrt(ud**2 + uq**2)
            I[qq, dd] = np.sqrt(id_val**2 + iq_val**2)
            T[qq, dd] = 1.5 * p * (psid * iq_val - psiq * id_val)

    Psi = np.sqrt(Psi_d**2 + Psi_q**2)

    # Plot results
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_xlabel("id")
    ax.set_ylabel("iq")

    ax.scatter(jnp.squeeze(physical_i_d), jnp.squeeze(physical_i_q), s=1)

    umult = 1 / jnp.sqrt(3)  # 2 / 3  # * jnp.sqrt(2)
    ax.contour(Id, Iq, I, levels=[250], colors="k", linewidths=1.5)
    if show_torque:
        ax.contour(Id, Iq, T, linestyles="dashed", colors="k")
    ax.contour(Id, Iq, U, levels=[udc * umult], colors="k", linewidths=1.5)

    ax.set_ylim(-350, 350)
    ax.set_xlim((-350, 200))

    return fig, ax
