import json
import datetime

import numpy as np
import jax

jax.config.update("jax_platform_name", "cpu")

from exciting_exciting_systems.related_work.np_reimpl.pendulum import Pendulum
from exciting_exciting_systems.related_work.algorithms import excite_with_GOATS


def identity(x):
    return x


def safe_json_dump(obj, fp):
    default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
    return json.dump(obj, fp, default=default)


### Start Experiment parameters
env_params = dict(batch_size=1, tau=2e-2, max_torque=8, g=9.81, l=1, m=1)
env = Pendulum(
    batch_size=env_params["batch_size"],
    max_torque=env_params["max_torque"],
    g=env_params["g"],
    l=env_params["l"],
    m=env_params["m"],
    tau=env_params["tau"],
)

alg_params = dict(
    n_amplitudes=100,
    bounds_duration=(1, 50),
    population_size=50,
    n_generations=300,
    featurize=identity,
)

seeds = [124, 2]
### End Experiment parameters

for seed in seeds:
    exp_params = dict(
        seed=seed,
        alg_params=alg_params,
        env_params=env_params,
    )

    # setup PRNG
    rng = np.random.default_rng(seed=seed)

    # save parameters
    file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    with open(f"../results/goats/params_{file_name}.json", "w") as fp:
        safe_json_dump(exp_params, fp)

    # run excitation algorithm
    observations, actions = excite_with_GOATS(
        n_amplitudes=alg_params["n_amplitudes"],
        env=env,
        bounds_duration=alg_params["bounds_duration"],
        population_size=alg_params["population_size"],
        n_generations=alg_params["n_generations"],
        featurize=identity,
        rng=np.random.default_rng(seed=exp_params["seed"]),
        verbose=True,
    )
    # save observations + actions
    with open(f"../results/goats/data_{file_name}.json", "w") as fp:
        json.dump(dict(observations=observations.tolist(), actions=actions.tolist()), fp)

    jax.clear_caches()
