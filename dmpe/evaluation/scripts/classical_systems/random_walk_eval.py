import json
import argparse
import datetime
import os
import pathlib

import numpy as np
import jax

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


from dmpe.data_management import DataPaths
from dmpe.utils.env_utils.fluid_tank_utils import setup_env as setup_fluid_tank_env
from dmpe.utils.env_utils.pendulum_utils import setup_env as setup_pendulum_env
from dmpe.utils.env_utils.cart_pole_utils import setup_env as setup_cart_pole_env
from dmpe.related_work.random_walk import excite_with_random_walk


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
    env, penalty_function, env_params = setup_pendulum_env()

elif sys_name == "fluid_tank":
    env, penalty_function, env_params = setup_fluid_tank_env()

elif sys_name == "cart_pole":
    env, penalty_function, env_params = setup_cart_pole_env()

exp_params = dict(
    seed=None,
    n_time_steps=15_000,
    env_params=env_params,
    alg_params=dict(
        n_tries=4000,
        penalty_function=penalty_function,
    ),
)

seeds = list(np.arange(101, 131))
### End experiment parameters #########################################################################################


### Start experiments #################################################################################################

for exp_idx, seed in enumerate(seeds):
    print("Running experiment", exp_idx, f"(seed: {seed}) on '{sys_name}'")
    exp_params["seed"] = int(seed)

    # Check that the targeted data folder actually exist:
    results_path = DataPaths().cs_experiments / pathlib.Path("random_walk") / pathlib.Path(sys_name)
    print(f"Results will be written to: '{results_path}'.")
    assert results_path.exists(), (
        f"The expected results path '{results_path}' does not seem to exist. Please create the necessary file structure "
        + "or adapt the path."
    )

    # setup PRNG
    key = jax.random.PRNGKey(seed=exp_params["seed"])

    # run excitation
    observations, actions = excite_with_random_walk(env, exp_params, key)

    # save results
    file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(results_path / pathlib.Path(f"params_{file_name}.json"), "w") as fp:
        safe_json_dump(exp_params, fp)

    # save observations + actions
    with open(results_path / pathlib.Path(f"data_{file_name}.json"), "w") as fp:
        json.dump(dict(observations=observations.tolist(), actions=actions.tolist()), fp)

    jax.clear_caches()

### End experiments ###################################################################################################
