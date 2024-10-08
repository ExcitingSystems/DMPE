# Differentiable Model Predictive Excitation (DMPE):

This repository implements an algorithm for the excitation of systems with unknown (usually non-linear) systems.
The inner workings and lines of thought are outlined within the corresponding publication.
If you found this repository useful for your research, please cite it as:

```
@Article{Vater**TBD**,
  author  = {**TBD**},
  journal = {**TBD**},
  title   = {**TBD**},
  year    = {**TBD**},
}
```

## Installation:

Simplest way, using `Python >= 3.11`:

```
pip install dmpe
```
- Intended for a Linux system using an NVIDIA GPU where CUDA is set up
- Theoretically, it can be used without a GPU and also on Windows, **but** performance will likely be suboptimal
- Depends on [`exciting_environments`](https://github.com/ExcitingSystems/exciting-environments)
- As of now, the requirements/dependencies are strict. It is likely that other versions work as well, but the given setup has been used extensively. (The requirements will likely be extended in the future.)
- As this repository is actively being worked on, it is possible that a more recent version is accessible in the [`DMPE`](https://github.com/ExcitingSystems/dmpe) GitHub repository.


**Alternative installation:**

Download the current state of the [`exciting_environments`](https://github.com/ExcitingSystems/exciting-environments) repository, e.g.:
```
git clone git@github.com:ExcitingSystems/exciting-environments.git
```
and install it in your python environment by moving to the downloaded folder and running `pip install .`.
Then, download the [`DMPE`](https://github.com/ExcitingSystems/dmpe) source code, e.g.:

```
git clone git@github.com:ExcitingSystems/DMPE.git
```

Afterwards, install it from within the repository folder via `pip install -e .` for an editable version or with `pip install .` if you do not plan to make changes to the code.


## Structure:

The repository is structured as follows:

- `dmpe/` contains the source code for the DMPE algorithm and also for the GOATS algorithms from the related work.
- `eval/` contains the code used in the experiments in the corresponding publication [Vater2024]. 
- `dev/` contains jupyter notebooks that are intended for development on the repository.
- `examples/` contains some examples to get started
- `fig/` contains example images (e.g., for the README)


## Basic Usage:

To apply the algorithms onto a system, the systems structure must comply to a specific API (Naturally, this can be adapted in the future. Please open an issue or write an e-mail to vater@lea.uni-paderborn.de, if you are interested in discussing this). Example environments following this API can be found in the [`exciting_environments`](https://github.com/ExcitingSystems/exciting-environments) repository.

Using the algorithm for such an environment is as simple as:

```py
import jax.numpy as jnp
import diffrax

import exciting_environments as excenvs
from dmpe.models.models import NeuralEulerODEPendulum
from dmpe.algorithms import excite_with_dmpe
from dmpe.algorithms.algorithm_utils import default_dmpe_parameterization


env = excenvs.make(
    env_id="Pendulum-v0",
    batch_size=1,
    action_constraints={"torque": 5},
    static_params={"g": 9.81, "l": 1, "m": 1},
    solver=diffrax.Tsit5(),
    tau=2e-2,
)

def featurize_theta(obs):
    """Transform angle information with sin() and cos()."""
    feat_obs = jnp.stack([jnp.sin(obs[..., 0] * jnp.pi), jnp.cos(obs[..., 0] * jnp.pi), obs[..., 1]], axis=-1)
    return feat_obs

# get default parameterization
exp_params, proposed_actions, loader_key, expl_key = default_dmpe_parameterization(
    env, seed=0, featurize=featurize_theta, model_class=NeuralEulerODEPendulum
)
exp_params["n_time_steps"] = 1000  # reduce to N=1000 steps

# run excitation
observations, actions, model, density_estimate, losses, proposed_actions = excite_with_dmpe(
    env,
    exp_params,
    proposed_actions,
    loader_key,
    expl_key,
)

# visualize
from dmpe.evaluation.plotting_utils import plot_sequence
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams.update({'font.size': 10})
mpl.rcParams['text.latex.preamble']=r"\usepackage{bm}\usepackage{amsmath}"

fig = plot_sequence(observations, actions, env.tau, env.obs_description, env.action_description)
plt.show()
```
![](https://github.com/ExcitingSystems/DMPE/blob/main/fig/simple_example_pendulum.png?raw=true)
