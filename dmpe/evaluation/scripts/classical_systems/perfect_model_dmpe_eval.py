import json
import datetime
import argparse
import os
import pathlib

import numpy as np
import jax
import jax.numpy as jnp

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import diffrax
import optax
from haiku import PRNGSequence

import exciting_environments as excenvs

from dmpe.utils.signals import aprbs
from dmpe.utils.density_estimation import select_bandwidth, get_uniform_target_distribution
from dmpe.excitation.excitation_utils import soft_penalty
from dmpe.algorithms.algorithms import excite_with_dmpe


# file path setup
REPO_ROOT_PATH = pathlib.Path(__file__).parent.parent.parent.parent.parent
TARGETED_DATA_PATH = REPO_ROOT_PATH / pathlib.Path("data") / pathlib.Path("classical_systems")


def safe_json_dump(obj, fp):
    default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
    return json.dump(obj, fp, default=default)


parser = argparse.ArgumentParser(description="Process 'sys_name' to choose the system to experiment on.")
parser.add_argument(
    "sys_name",
    metavar="sys_name",
    type=str,
    help="The name of the environment. Options are ['pendulum', 'fluid_tank', 'cart_pole'].",
)
parser.add_argument("--gpu_id", type=int, default=0, help="GPU id to use.")

args = parser.parse_args()
sys_name = args.sys_name

gpus = jax.devices()
gpu_id = args.gpu_id
jax.config.update("jax_default_device", gpus[args.gpu_id])

### Start experiment parameters #######################################################################################
if sys_name == "pendulum":
    ## Start pendulum experiment parameters

    env_params = dict(batch_size=1, tau=2e-2, max_torque=5, g=9.81, l=1, m=1, env_solver=diffrax.Tsit5())
    env = excenvs.make(
        env_id="Pendulum-v0",
        batch_size=env_params["batch_size"],
        action_normalizations={
            "torque": excenvs.utils.MinMaxNormalization(min=-env_params["max_torque"], max=env_params["max_torque"])
        },
        static_params={"g": env_params["g"], "l": env_params["l"], "m": env_params["m"]},
        solver=env_params["env_solver"],
        tau=env_params["tau"],
    )
    alg_params = dict(
        bandwidth=None,
        n_prediction_steps=20,
        points_per_dim=21,
        grid_extend=1.05,
        excitation_optimizer=optax.adabelief(1e-1),
        n_opt_steps=10,
        start_optimizing=5,
        consider_action_distribution=True,
        penalty_function=None,
        target_distribution=None,
        clip_action=True,
        n_starts=5,
        reuse_proposed_actions=True,
    )

    alg_params["bandwidth"] = float(
        select_bandwidth(
            delta_z=2,
            dim=env.physical_state_dim + env.action_dim,
            n_g=alg_params["points_per_dim"],
            percentage=0.3,
        )
    )

    # overwrite penalty function and target distribution
    alg_params["penalty_function"] = lambda x, u: soft_penalty(a=x, a_max=1, penalty_order=2) + soft_penalty(
        a=u, a_max=1, penalty_order=2
    )
    alg_params["target_distribution"] = get_uniform_target_distribution(
        dim=3 if alg_params["consider_action_distribution"] else 2,
        points_per_dim=alg_params["points_per_dim"],
        bandwidth=alg_params["bandwidth"],
        grid_extend=alg_params["grid_extend"],
        consider_action_distribution=alg_params["consider_action_distribution"],
        penalty_function=alg_params["penalty_function"],
    )

    exp_params = dict(
        seed=None,
        n_time_steps=15_000,
        model_class=None,
        env_params=env_params,
        alg_params=alg_params,
        model_trainer_params=None,
        model_params=None,
    )
    seeds = list(np.arange(101, 131))
    ## End pendulum experiment parameters

elif sys_name == "fluid_tank":
    ## Start fluid_tank experiment parameters

    env_params = dict(
        batch_size=1,
        tau=5,
        max_height=3,
        max_inflow=0.2,
        base_area=jnp.pi,
        orifice_area=jnp.pi * 0.1**2,
        c_d=0.6,
        g=9.81,
        env_solver=diffrax.Tsit5(),
    )
    env = excenvs.make(
        "FluidTank-v0",
        physical_normalizations=dict(height=excenvs.utils.MinMaxNormalization(min=0, max=env_params["max_height"])),
        action_normalizations=dict(inflow=excenvs.utils.MinMaxNormalization(min=0, max=env_params["max_inflow"])),
        static_params=dict(
            base_area=env_params["base_area"],
            orifice_area=env_params["orifice_area"],
            c_d=env_params["c_d"],
            g=env_params["g"],
        ),
        tau=env_params["tau"],
        solver=env_params["env_solver"],
    )

    alg_params = dict(
        bandwidth=None,
        n_prediction_steps=10,
        points_per_dim=50,
        grid_extend=1.05,
        excitation_optimizer=optax.adabelief(1e-1),
        n_opt_steps=10,
        start_optimizing=5,
        consider_action_distribution=True,
        penalty_function=None,
        target_distribution=None,
        clip_action=True,
        n_starts=5,
        reuse_proposed_actions=True,
    )

    alg_params["bandwidth"] = float(
        select_bandwidth(
            delta_z=2,
            dim=env.physical_state_dim + env.action_dim,
            n_g=alg_params["points_per_dim"],
            percentage=0.3,
        )
    )

    # overwrite penalty function and target distribution
    alg_params["penalty_function"] = lambda x, u: soft_penalty(a=x, a_max=1, penalty_order=2) + soft_penalty(
        a=u, a_max=1, penalty_order=2
    )
    alg_params["target_distribution"] = get_uniform_target_distribution(
        dim=2 if alg_params["consider_action_distribution"] else 1,
        points_per_dim=alg_params["points_per_dim"],
        bandwidth=alg_params["bandwidth"],
        grid_extend=alg_params["grid_extend"],
        consider_action_distribution=alg_params["consider_action_distribution"],
        penalty_function=alg_params["penalty_function"],
    )

    exp_params = dict(
        seed=None,
        n_time_steps=15_000,
        model_class=None,
        env_params=env_params,
        alg_params=alg_params,
        model_trainer_params=None,
        model_params=None,
    )
    seeds = list(np.arange(101, 131))
    ## End fluid_tank experiment parameters

elif sys_name == "cart_pole":
    ## Start cart_pole experiment parameters

    env_params = dict(
        batch_size=1,
        tau=2e-2,
        max_force=10,
        static_params={
            "mu_p": 0.002,
            "mu_c": 0.5,
            "l": 0.5,
            "m_p": 0.1,
            "m_c": 1,
            "g": 9.81,
        },
        physical_normalizations={
            "deflection": excenvs.utils.MinMaxNormalization(min=-2.4, max=2.4),
            "velocity": excenvs.utils.MinMaxNormalization(min=-8, max=8),
            "theta": excenvs.utils.MinMaxNormalization(min=-jnp.pi, max=jnp.pi),
            "omega": excenvs.utils.MinMaxNormalization(min=-8, max=8),
        },
        env_solver=diffrax.Tsit5(),
    )
    env = excenvs.make(
        env_id="CartPole-v0",
        batch_size=env_params["batch_size"],
        action_normalizations={
            "force": excenvs.utils.MinMaxNormalization(min=-env_params["max_force"], max=env_params["max_force"])
        },
        physical_normalizations=env_params["physical_normalizations"],
        static_params=env_params["static_params"],
        solver=env_params["env_solver"],
        tau=env_params["tau"],
    )

    alg_params = dict(
        bandwidth=0.12,
        n_prediction_steps=50,
        points_per_dim=10,
        grid_extend=1.05,
        excitation_optimizer=optax.adabelief(1e-1),
        n_opt_steps=5,
        start_optimizing=5,
        consider_action_distribution=True,
        penalty_function=None,
        target_distribution=None,
        clip_action=True,
        n_starts=5,
        reuse_proposed_actions=True,
    )

    # alg_params["bandwidth"] = float(
    #     select_bandwidth(
    #         delta_z=2,
    #         dim=env.physical_state_dim + env.action_dim,
    #         n_g=alg_params["points_per_dim"],
    #         percentage=0.1,
    #     )
    # )

    # overwrite penalty function and target distribution
    alg_params["penalty_function"] = lambda x, u: soft_penalty(a=x, a_max=1, penalty_order=2) + soft_penalty(
        a=u, a_max=1, penalty_order=2
    )
    alg_params["target_distribution"] = get_uniform_target_distribution(
        dim=5 if alg_params["consider_action_distribution"] else 4,
        points_per_dim=alg_params["points_per_dim"],
        bandwidth=alg_params["bandwidth"],
        grid_extend=alg_params["grid_extend"],
        consider_action_distribution=alg_params["consider_action_distribution"],
        penalty_function=alg_params["penalty_function"],
    )

    exp_params = dict(
        seed=None,
        n_time_steps=15_000,
        model_class=None,
        env_params=env_params,
        alg_params=alg_params,
        model_trainer_params=None,
        model_params=None,
    )
    seeds = list(np.arange(101, 131))

    ## End cart_pole experiment parameters

elif sys_name == "pmsm":
    ## Begin pmsm experiment parameters
    from dmpe.excitation.excitation_utils import soft_penalty
    from exciting_environments.pmsm import PMSM

    class ExcitingPMSM(PMSM):

        def generate_observation(self, system_state, env_properties):
            physical_constraints = env_properties.physical_constraints

            eps = system_state.physical_state.epsilon
            cos_eps = jnp.cos(eps)
            sin_eps = jnp.sin(eps)

            obs = jnp.hstack(
                (
                    (system_state.physical_state.i_d + (physical_constraints.i_d * 0.5))
                    / (physical_constraints.i_d * 0.5),
                    system_state.physical_state.i_q / physical_constraints.i_q,
                )
            )
            return obs

        def init_state(self, env_properties, rng=None, vmap_helper=None):
            """Returns default initial state for all batches."""
            phys = self.PhysicalState(
                u_d_buffer=0.0,
                u_q_buffer=0.0,
                epsilon=0.0,
                i_d=-env_properties.physical_constraints.i_d / 2,
                i_q=0.0,
                torque=0.0,
                omega_el=2 * jnp.pi * 3 * 1000 / 60,
            )
            subkey = jnp.nan
            additions = None  # self.Optional(something=jnp.zeros(self.batch_size))
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

    batch_size = 1

    env = ExcitingPMSM(
        batch_size=batch_size,
        saturated=True,
        static_params={
            "p": 3,
            "r_s": 15e-3,
            "l_d": jnp.nan,
            "l_q": jnp.nan,
            "psi_p": jnp.nan,
            "deadtime": 0,
        },
        solver=diffrax.Euler(),
    )

    def PMSM_penalty(observations, actions, penalty_order=2):

        action_penalty = soft_penalty(actions, a_max=1, penalty_order=1)

        physical_i_d = observations[..., 0] * (env.env_properties.physical_constraints.i_d * 0.5) - (
            env.env_properties.physical_constraints.i_d * 0.5
        )
        physical_i_q = observations[..., 1] * env.env_properties.physical_constraints.i_q

        a = physical_i_d / 250
        b = physical_i_q / 250

        obs_penalty = jax.nn.relu(a**2 + b**2 - 0.9)
        obs_penalty = jnp.sum(obs_penalty)
        i_d_penalty = jnp.sum(jax.nn.relu(a))

        return (obs_penalty + i_d_penalty + action_penalty) * 1e3

    alg_params = dict(
        bandwidth=jnp.nan,
        n_prediction_steps=5,
        points_per_dim=21,
        action_lr=1e-2,
        n_opt_steps=100,
        consider_action_distribution=True,
        penalty_function=PMSM_penalty,
        target_distribution=None,
        clip_action=False,
        n_starts=3,
        reuse_proposed_actions=True,
    )

    dim = 4 if alg_params["consider_action_distribution"] else 2
    points_per_dim = alg_params["points_per_dim"]
    target_distribution = (np.ones(shape=[points_per_dim for _ in range(dim)]) ** dim)[..., None]
    xx, yy = np.meshgrid(np.linspace(-1, 0, points_per_dim), np.linspace(-1, 1, points_per_dim))
    target_distribution[xx**2 + yy**2 > 1] = 0
    target_distribution = target_distribution / jnp.sum(target_distribution)
    alg_params["target_distribution"] = jnp.array(target_distribution.reshape(-1, 1))

    alg_params["bandwidth"] = float(
        select_bandwidth(
            delta_z=2,
            dim=dim,
            n_g=alg_params["points_per_dim"],
            percentage=0.3,
        )
    )

    exp_params = dict(
        seed=int(1),
        n_time_steps=5_000,
        model_class=None,
        env_params=None,
        alg_params=alg_params,
        model_trainer_params=None,
        model_params=None,
    )

    seeds = list(np.arange(22, 32))
    ## End pmsm experiment parameters

else:
    raise NotImplementedError(f"System '{sys_name}' is unknown. Choose from ['pendulum', 'fluid_tank', 'cart_pole'].")

### End experiment parameters #########################################################################################

for exp_idx, seed in enumerate(seeds):
    print("Running experiment", exp_idx, f"(seed: {seed}) on '{sys_name}'")
    exp_params["seed"] = int(seed)

    # Check that the targeted data folder actually exist:
    results_path = TARGETED_DATA_PATH / pathlib.Path("perfect_model_dmpe") / pathlib.Path(sys_name)
    print(f"Results will be written to: '{results_path}'.")
    assert results_path.exists(), (
        f"The expected results path '{results_path}' does not seem to exist. Please create the necessary file structure "
        + "or adapt the path."
    )

    # setup PRNG
    key = jax.random.PRNGKey(seed=exp_params["seed"])
    data_key, _, _, expl_key, key = jax.random.split(key, 5)
    data_rng = PRNGSequence(data_key)

    # initial guess
    proposed_actions = jnp.hstack(
        [
            aprbs(alg_params["n_prediction_steps"], env.batch_size, 1, 10, next(data_rng))[0]
            for _ in range(env.action_dim)
        ]
    )
    # run excitation algorithm
    observations, actions, model, density_estimate, losses, proposed_actions = excite_with_dmpe(
        env,
        exp_params,
        proposed_actions,
        None,
        expl_key,
    )

    # save parameters
    file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(TARGETED_DATA_PATH / pathlib.Path(f"params_{file_name}.json"), "w") as fp:
        safe_json_dump(exp_params, fp)

    # save observations + actions
    with open(TARGETED_DATA_PATH / pathlib.Path(f"data_{file_name}.json"), "w") as fp:
        json.dump(dict(observations=observations.tolist(), actions=actions.tolist()), fp)

    jax.clear_caches()

### End experiments ###################################################################################################
