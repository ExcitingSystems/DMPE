import glob
import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import jax.numpy as jnp
import equinox as eqx

from dmpe.evaluation.exp_data_model_learning import ModelExpDataResult


def plot_jsd_model_relation(data_path: pathlib.Path, model_class: eqx.Module, verbose: bool = False):
    means = []
    medians = []
    jsds = []
    colors = []

    color_cycle = plt.rcParams["axes.prop_cycle"]()
    color_mapping = [next(color_cycle)["color"] for _ in range(15)]

    result_paths = glob.glob(str(data_path / "*.eqx"))
    n_results = len(result_paths)
    print("# or results:", n_results)
    print(80 * "-")

    for result_path in result_paths:
        result = ModelExpDataResult.from_file(
            filename=result_path,
            model_class=model_class,
        )
        color_idx = int(result.n_datapoints / 1_000) - 1
        colors.append(color_mapping[color_idx])

        means.append(jnp.mean(jnp.array(result.model_errors), axis=0)[-1])
        medians.append(jnp.median(jnp.array(result.model_errors), axis=0)[-1])
        jsds.append(result.data_jsd)

        if verbose:
            print(result.data_jsd)
            fig, _ = result.visualize()
            plt.show()
            print(80 * "-")

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.grid(True)

    ax.scatter(
        jsds, medians, s=25, marker="x", c=colors
    )  # , c=next(colors)["color"], label=f"{data_length} data points")

    ax.set_ylabel("model prediction loss")
    ax.set_xlabel("JSD")
    # ax.set_ylabel(r"$\mathcal{L}_{\mathcal{M}}$")
    # ax.set_xlabel(r"$\mathcal{L}_{\mathrm{JSD}}$")
    ax.grid(True)
    ax.set_yscale("log")
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="x",
            color="w",
            label=(idx + 1) * 1000,
            markerfacecolor=color_mapping[idx],
            markeredgecolor=color_mapping[idx],
            markersize=5,
            linestyle="None",
        )
        for idx in range(len(color_mapping))
    ]
    # legend_elements = [Patch(facecolor=color_mapping[idx], label=(idx + 1) * 1000) for idx in range(len(color_mapping))]
    ax.legend(handles=legend_elements, title=r"\# of datapoints")

    return fig, ax
