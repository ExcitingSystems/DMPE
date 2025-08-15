from uuid import uuid4
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp

from dmpe.data_management import DataPaths
from dmpe.utils.env_utils.cart_pole_utils import setup_env as setup_cart_pole_env
from dmpe.utils.sets.reachable_set import approximate_reachable_set_through_chunks
from dmpe.utils.sets.shared import save_discretized_set


env, penalty_function, featurize, _ = setup_cart_pole_env()
key = jax.random.PRNGKey(14)

obs_dim = env.reset(env.env_properties)[0].shape[-1]
sequence_length = 200
n_starts = 10
n_opt_steps = 100  # 50_000
tolerance = 1e-4
points_per_dim = 10
chunk_size = 20_000

xs = [
    jnp.linspace(-1.0, 1.0, 25),
    jnp.linspace(-1.0, 1.0, 25),
    jnp.linspace(-1.0, 1.0, 25),
    jnp.linspace(-1.0, 1.0, 25),
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
save_discretized_set(
    DataPaths().reach_ci_experiments / f"cart_pole_Rs_{exp_id}.json",
    set=R_s,
)
