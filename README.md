# Differentiable Model Predictive Excitation (DMPE):

This repository implements an algorithm for the excitation of systems with unknown (usually non-linear) dynamics.
The inner workings and lines of thought are outlined within the corresponding publication.
If you found this repository useful for your research, please cite the current preprint version as:

```
@Article{Vater2024,
  author      = {Vater, Hendrik and Wallscheid, Oliver},
  title       = {Differentiable Model Predictive Excitation: Generating Optimal Data Sets for Learning of Dynamical System Models},
  journal     = {TechRxiv preprint},
  year        = {2024},
  doi         = {10.36227/techrxiv.172840381.16440835/v1},
}
```

## Installation:

> If you are specifically interested in reproducing the results from the `Vater2024` publication, you are kindly referred to: https://pypi.org/project/dmpe/0.1.3/

> Else, if you are specifically interested in reproducing the results from the `TBD` related publication, you are kindly referred to: https://pypi.org/project/dmpe/0.2.1/

Otherwise, the simplest way is using `Python >= 3.11`:

```
pip install dmpe
```
- Intended for a Linux system using an NVIDIA GPU where CUDA is set up
- Theoretically, it can be used without a GPU and also on Windows, **but** performance will likely be suboptimal and the results are not exactly reproducible with the GPU/Linux results.
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

- `data/` is used to store the experiment results (can be created manually or via a script located at `evaluation/scripts/create_exp_directories.py`) 
- `dmpe/` contains the **source code for the DMPE algorithm**, for the GOATS algorithms from the related work, and **the scripts to run experiments**.
- `examples/` contains some examples to get started that **are regularly updated to reflect the momentary state of the repo.**
- `fig/` contains example images (e.g., for the README)
- `notebooks/eval` contains jupyter notebooks that are intended for evaluation of experiments (generally not maintained and only updated when needed). 
- `notebooks/dev/` contains jupyter notebooks that are intended for development on the repository (generally not maintained and only updated when needed).


## Basic Usage:

To apply the algorithms onto a system, the systems structure must comply to a specific API (Naturally, this can be adapted in the future. Please open an issue or write an e-mail to vater@lea.uni-paderborn.de, if you are interested in discussing this). Example environments following this API can be found in the [`exciting_environments`](https://github.com/ExcitingSystems/exciting-environments) repository.

Using the algorithm for such an environment is as simple as:

```py
import jax.numpy as jnp
import diffrax

import exciting_environments as excenvs
from dmpe.models.models import NeuralEulerODEPendulum
from dmpe.algorithms.algorithms import excite_with_dmpe
from dmpe.algorithms.algorithm_utils import default_dmpe_parameterization


env = excenvs.make(
    env_id="Pendulum-v0",
    batch_size=1,
    action_normalizations = {"torque": excenvs.utils.MinMaxNormalization(min=-5, max=5)},
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
exp_params["n_time_steps"] = 1500  # reduce number of timesteps to N=1500

# run excitation
observations, actions, model, density_estimate, losses, proposed_actions, _ = excite_with_dmpe(
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


## Further Examples:

Additional examples can be found in the `examples/` folder.
There, the `CartPole` and `PMSM` are excited using `DMPE`.

Exemplary `PMSM` results at $n = 5000 \, \mathrm{min}^{-1}$:

- Exemplary acquired trajectory (without considering the action distribution):

![](https://github.com/ExcitingSystems/DMPE/blob/main/fig/MIMO_example_trajectories_no_action_coverage.png?raw=true)

- feature-space coverage of the resulting data (without considering the action distribution):

![](https://github.com/ExcitingSystems/DMPE/blob/main/fig/MIMO_example_coverage_no_action_coverage.png?raw=true)
