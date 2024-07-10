"""Pure numpy implementation of the pendulum env from an earlier version of exciting-environments.

This is necessary for a pure numpy implementation of the related work (which cannot be efficiently implemented in jax).
"""

import numpy as np
from gymnasium import spaces
from gymnasium import vector


class Pendulum:
    """
    Description:
        Environment to simulate a simple Pendulum.

    State Variables:
        ``['theta' , 'omega']''``

    Action Variable:
        ``['torque']''``

    Observation Space (State Space):
        Box(low=[-1, -1], high=[1, 1])

    Action Space:
        Box(low=-1, high=1)

    Initial State:
        Unless chosen otherwise, theta equals 1(normalized to pi) and omega is set to zero.

    Example:
        >>> #TODO

    """

    def __init__(self, l=1, m=1, max_torque=20, reward_func=None, g=9.81, tau=1e-4, constraints=[10]):
        """
        Args:
            l(float): Length of the pendulum. Default: 1
            m(float): Mass of the pendulum tip. Default: 1
            max_torque(float): Maximum torque that can be applied to the system as action. Default: 20
            reward_func(function): Reward function for training. Needs Observation-Matrix and Action as Parameters.
                                    Default: None (default_reward_func from class)
            g(float): Gravitational acceleration. Default: 9.81
            tau(float): Duration of one control step in seconds. Default: 1e-4.
            constraints(array): Constraints for state ['omega'] (array with length 1). Default: [10]

        """
        self.g = g
        self.tau = tau
        self.l = l
        self.m = m
        self.max_torque = max_torque

        self.state_normalizer = np.concatenate(([np.pi], constraints), axis=0)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), shape=(2,), dtype=np.float32
        )

        if reward_func:
            self.reward_func = reward_func
        else:
            self.reward_func = self.default_reward_func

    def _step(self, states_norm, torque_norm):

        torque = torque_norm * self.max_torque
        states = self.state_normalizer * states_norm
        theta = states[0]
        omega = states[1]

        dtheta = omega
        domega = (torque + self.l * self.m * self.g * np.sin(theta)) / (self.m * (self.l) ** 2)

        theta_k1 = theta + self.tau * dtheta  # explicit Euler
        theta_k1 = ((theta_k1 + np.pi) % (2 * np.pi)) - np.pi
        omega_k1 = omega + self.tau * domega  # explicit Euler

        states_k1 = np.hstack(
            (
                theta_k1,
                omega_k1,
            )
        )
        states_k1_norm = states_k1 / self.state_normalizer

        return states_k1_norm

    def generate_observation(self, states):
        return np.hstack(
            (
                states,
                # torque,
            )
        )

    def step(self, state, action, *args, **kwargs):
        """This function's API is bloated to fit to the exciting environments API."""
        state = self._step(state, action)
        return self.generate_observation(state), None, None, None, state

    @property
    def env_properties(self):
        """Necessary for exciting environments API, but not necessary for functionality"""
        return None

    @property
    def def_reward_func(self):
        return self.default_reward_func

    def default_reward_func(self, obs, action):
        return np.array((obs[0]) ** 2 + 0.1 * (obs[1]) ** 2 + 0.1 * (action[0]) ** 2)

    @property
    def obs_description(self):
        return self.states_description

    @property
    def states_description(self):
        return np.array(["theta", "omega"])

    @property
    def action_description(self):
        return np.array(["torque"])

    def reset(self, random_initial_values=False, initial_values: np.ndarray = None):

        self.states = np.array([1.0, 0.0])

        obs = self.generate_observation(self.states)

        return obs, self.states
