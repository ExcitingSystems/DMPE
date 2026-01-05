from copy import deepcopy
import json
import pathlib
from tqdm import tqdm
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix


def fit_params_on_experiment_data(
    key: jax.random.PRNGKey,
    observations: jax.Array,
    actions: jax.Array,
):
    raise NotImplementedError()
