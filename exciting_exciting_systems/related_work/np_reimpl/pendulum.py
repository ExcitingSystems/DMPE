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

    def __init__(self, batch_size=8, l=1, m=1, max_torque=20, reward_func=None, g=9.81, tau=1e-4, constraints=[10]):
        """
        Args:
            batch_size(int): Number of training examples utilized in one iteration. Default: 8
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
        self.l_const = l
        self.m_const = m
        self.max_torque = max_torque
        self.batch_size = batch_size

        self.state_normalizer = np.concatenate(([np.pi], constraints), axis=0)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.batch_size, 1), dtype=np.float32)

        # single_obs_space=spaces.Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([-1.0, 1.0 , 1.0]),shape=(3,) ,dtype=np.float32)
        single_obs_space = spaces.Box(
            low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), shape=(2,), dtype=np.float32
        )
        self.observation_space = vector.utils.batch_space(single_obs_space, n=batch_size)

        if reward_func:
            if self.test_rew_func(reward_func):
                self.reward_func = reward_func
        else:
            self.reward_func = self.default_reward_func

    def update_batch_dim(self):
        self.l = np.full((self.batch_size, 1), self.l_const)
        self.m = np.full((self.batch_size, 1), self.m_const)
        theta = np.full((self.batch_size), 1).reshape(-1, 1)
        omega = np.zeros(self.batch_size).reshape(-1, 1)
        self.states = np.hstack(
            (
                theta,
                omega,
            )
        )

    def test_rew_func(self, func):
        try:
            out = func(np.zeros([self.batch_size, int(len(self.get_obs_description()))]))
        except:
            raise Exception("Reward function should be using obs matrix as only parameter")
        try:
            if out.shape != (self.batch_size, 1):
                raise Exception("Reward function should be returning vector in shape (batch_size,1)")
        except:
            raise Exception("Reward function should be returning vector in shape (batch_size,1)")
        return True

    def step(self, states_norm, torque_norm):

        torque = torque_norm * self.max_torque
        states = self.state_normalizer * states_norm
        theta = states[:, 0].reshape(-1, 1)
        omega = states[:, 1].reshape(-1, 1)

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

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        # If batchsize change, update the corresponding dimension
        self._batch_size = batch_size
        self.update_batch_dim()

    def generate_observation(self, states):
        return np.hstack(
            (
                states,
                # torque,
            )
        )

    @property
    def def_reward_func(self):
        return self.default_reward_func

    def default_reward_func(self, obs, action):
        return ((obs[:, 0]) ** 2 + 0.1 * (obs[:, 1]) ** 2 + 0.1 * (action[:, 0]) ** 2).reshape(-1, 1)

    @property
    def obs_description(self):
        return self.states_description

    @property
    def states_description(self):
        return np.array(["theta", "omega"])

    @property
    def action_description(self):
        return np.array(["torque"])

    def render(self):
        raise NotImplementedError("To be implemented!")

    def close(self):
        raise NotImplementedError("To be implemented!")

    def reset(self, random_initial_values=False, initial_values: np.ndarray = None):
        if random_initial_values:
            self.states = self.observation_space.sample()
        elif initial_values != None:
            assert (
                initial_values.shape[0] == self.batch_size
            ), f"number of rows is expected to be batch_size, got: {initial_values.shape[0]}"
            assert initial_values.shape[1] == len(
                self.obs_description
            ), f"number of columns is expected to be amount obs_entries: {len(self.obs_description)}, got: {initial_values.shape[0]}"
            self.states = initial_values
        else:
            self.states[:, 0:1] = np.full(self.batch_size, 1).reshape(-1, 1)
            self.states[:, 1:2] = np.zeros(self.batch_size).reshape(-1, 1)

        obs = self.generate_observation(self.states)

        return obs, self.states
