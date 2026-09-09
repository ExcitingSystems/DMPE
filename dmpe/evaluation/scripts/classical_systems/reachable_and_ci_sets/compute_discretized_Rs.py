from uuid import uuid4
import argparse
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp

from dmpe.data_management import DataPaths
from dmpe.utils.env_utils.fluid_tank_utils import setup_env as setup_fluid_tank_env
from dmpe.utils.env_utils.pendulum_utils import setup_env as setup_pendulum_env
from dmpe.utils.env_utils.cart_pole_utils import setup_env as setup_cart_pole_env
from dmpe.utils.sets.reachable_set import approximate_reachable_set_through_chunks
from dmpe.utils.sets.shared import save_discretized_set


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

if sys_name == "fluid_tank":
    env, penalty_function, featurize, _ = setup_fluid_tank_env()
    key = jax.random.PRNGKey(14)

    obs_dim = env.reset()[0].shape[-1]
    sequence_length = 200
    n_starts = 100
    n_opt_steps = 50_000
    tolerance = 1e-4
    points_per_dim = 50
    chunk_size = 20_000

    xs = [
        jnp.linspace(-1.0, 1.0, points_per_dim),
    ]
    z_g = jnp.meshgrid(*xs, indexing="ij")
    z_g = jnp.stack([_x for _x in z_g], axis=-1)
    unflattened_shape = z_g.shape[:-1]
    target_observations_rs = z_g.reshape(-1, obs_dim)

elif sys_name == "pendulum":
    env, penalty_function, featurize, _ = setup_pendulum_env()
    key = jax.random.PRNGKey(14)

    obs_dim = env.reset()[0].shape[-1]
    sequence_length = 200
    n_starts = 400
    n_opt_steps = 50_000
    tolerance = 1e-4
    points_per_dim = 50
    chunk_size = 20_000

    xs = [
        jnp.linspace(-1.0, 1.0, points_per_dim),
        jnp.linspace(-1.0, 1.0, points_per_dim),
    ]
    z_g = jnp.meshgrid(*xs, indexing="ij")
    z_g = jnp.stack([_x for _x in z_g], axis=-1)
    unflattened_shape = z_g.shape[:-1]
    target_observations_rs = z_g.reshape(-1, obs_dim)

elif sys_name == "cart_pole":
    env, penalty_function, featurize, _ = setup_cart_pole_env()
    key = jax.random.PRNGKey(14)

    obs_dim = env.reset()[0].shape[-1]
    sequence_length = 200
    n_starts = 10
    n_opt_steps = 50_000
    tolerance = 1e-4
    points_per_dim = 25
    chunk_size = 20_000

    xs = [
        jnp.linspace(-1.0, 1.0, points_per_dim),
        jnp.linspace(-1.0, 1.0, points_per_dim),
        jnp.linspace(-1.0, 1.0, points_per_dim),
        jnp.linspace(-1.0, 1.0, points_per_dim),
    ]
    z_g = jnp.meshgrid(*xs, indexing="ij")
    z_g = jnp.stack([_x for _x in z_g], axis=-1)
    unflattened_shape = z_g.shape[:-1]
    target_observations_rs = z_g.reshape(-1, obs_dim)

R_s = approximate_reachable_set_through_chunks(
    env,
    chunk_size,
    target_observations_rs,
    penalty_function,
    featurize,
    key=key,
    sequence_length=sequence_length,
    n_starts=n_starts,
    n_opt_steps=n_opt_steps,
    tolerance=tolerance,
    unflattened_shape=unflattened_shape,
)


exp_id = str(uuid4())[:16]
file_path = DataPaths().reach_ci_experiments / f"{sys_name}_Rs_{exp_id}.json"
print(f"Set successfully computed. Storing set at {file_path}.")
save_discretized_set(
    file_path,
    set=R_s,
)
