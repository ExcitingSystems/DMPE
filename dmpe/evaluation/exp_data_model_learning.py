from copy import deepcopy
import json
import pathlib
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx

from dmpe.models.model_training import ModelTrainer
from dmpe.evaluation.model_evaluation import ModelEvaluator, NodeModelWrapper


def train_model_on_experiment_data(
    key: jax.random.PRNGKey,
    observations: jax.Array,
    actions: jax.Array,
    model_trainer_params: dict,
    model_params: dict,
    n_iters: int,
    model_class: eqx.Module,
    model_evaluator: ModelEvaluator,
) -> tuple[eqx.Module, list[float]]:
    """Trains a model on the provided experiment data.

    Args:
        key (jax.random.PRNGKey): Random key for sampling of trainng data
        observations (jax.Array): Observations from the experiment
        actions (jax.Array): Actions taken during the experiment
        model_trainer_params (dict): Parameters for the model trainer
        model_params (dict): Parameters for the model to be trained
        n_iters (int): Number of fitting iterations. The ModelTrainer does multiple
            training steps per iteration
        model_class (eqx.Module): The class of the model to be trained
        model_evaluator (ModelEvaluator): The evaluator for the model. Used to compare
            the model performance with the underlying ground-truth system from which the
            data was generated

    Returns:
        model (model_class): The trained model
        model_errors_log (list[float]): List of model errors after each fitting iteration
    """
    key, model_key, loader_key = jax.random.split(key, 3)
    model_trainer = ModelTrainer(**model_trainer_params)
    model_params = deepcopy(model_params)
    model_params["key"] = model_key

    model = model_class(**model_params)
    opt_state_model = model_trainer.model_optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    model_errors_log = []

    for iteration in tqdm(range(n_iters)):
        model, opt_state_model, loader_key = model_trainer.fit(
            model=model,
            k=jnp.array([observations.shape[0]]),
            observations=observations,
            actions=actions,
            opt_state=opt_state_model,
            loader_key=loader_key,
        )

        _, metric = model_evaluator.default_metrics["pred_comp"](
            NodeModelWrapper(model, featurize=model_trainer.featurize), model_evaluator.gt_model
        )
        model_errors_log.append(metric)

    return model, model_errors_log


class ModelExpDataResult(eqx.Module):
    exp_id: str
    seeds: list[int]
    n_datapoints: int
    n_obs: int
    n_actions: int
    observations: jax.Array
    actions: jax.Array
    data_jsd: float
    model_params: dict
    model_class: eqx.Module
    models: list[eqx.Module]
    n_iters: int
    model_errors: list[float]

    def save_to_file(self, file_path: str | pathlib.Path):

        hyperparams = dict(
            exp_id=self.exp_id,
            seeds=self.seeds,
            n_datapoints=self.n_datapoints,
            n_obs=self.n_obs,
            n_actions=self.n_actions,
            n_iters=self.n_iters,
            model_params=self.model_params,
        )

        with open(file_path, "wb") as f:
            hyperparam_str = json.dumps(hyperparams)
            f.write((hyperparam_str + "\n").encode())
            eqx.tree_serialise_leaves(f, self)

    @classmethod
    def from_file(
        cls,
        filename: str | pathlib.Path,
        model_class: eqx.Module,
    ) -> "ModelExpDataResult":
        with open(filename, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            result = cls.from_hyperparams(**hyperparams, model_class=model_class)
            return eqx.tree_deserialise_leaves(f, result)

    @classmethod
    def from_data(
        cls,
        exp_id: str,
        seeds: list[str],
        observations: jax.Array,
        actions: jax.Array,
        data_jsd: jax.Array,
        model_params: dict,
        model_class: eqx.Module,
        models: list[eqx.Module],
        model_errors: jax.Array,
    ) -> "ModelExpDataResult":
        assert observations.shape[0] == actions.shape[0]

        return cls(
            exp_id=exp_id,
            seeds=seeds,
            n_datapoints=observations.shape[0],
            n_obs=observations.shape[-1],
            n_actions=actions.shape[-1],
            observations=observations,
            actions=actions,
            data_jsd=data_jsd,
            model_params=model_params,
            model_class=model_class,
            models=models,
            n_iters=model_errors.shape[-1],
            model_errors=model_errors,
        )

    @classmethod
    def from_hyperparams(
        cls,
        exp_id: str,
        seeds: list[int],
        n_datapoints: int,
        n_obs: int,
        n_actions: int,
        n_iters: int,
        model_params: dict,
        model_class: eqx.Module,
    ) -> "ModelExpDataResult":
        return cls(
            exp_id=exp_id,
            seeds=seeds,
            n_datapoints=n_datapoints,
            n_obs=n_obs,
            n_actions=n_actions,
            observations=jnp.zeros((n_datapoints, n_obs)),
            actions=jnp.zeros((n_datapoints, n_actions)),
            data_jsd=jnp.zeros(shape=()),
            model_params=model_params,
            model_class=model_class,
            models=[model_class(**model_params, key=jax.random.PRNGKey(0)) for _ in seeds],
            n_iters=n_iters,
            model_errors=jnp.zeros((len(seeds), n_iters)),
        )

    def visualize_model_prediction_performance(self, wrapped_model, model_evaluator: ModelEvaluator, labels: list[str]):
        difference_map, _ = model_evaluator.default_metrics["pred_comp"](
            wrapped_model,
            model_evaluator.gt_model,
        )

        n_features = model_evaluator.obs_dim + model_evaluator.act_dim
        reshaped_difference_map = difference_map.reshape(
            [model_evaluator.validation_points_per_dim] * n_features + [-1]
        )
        # abs_map = jnp.mean(jnp.abs(reshaped_difference_map) ** 2, axis=-1)
        abs_map = jnp.linalg.norm(reshaped_difference_map, axis=-1)

        fig, axs = plt.subplots(nrows=n_features, ncols=n_features, figsize=(9, 9), sharex=True, sharey=True)

        feature_indices = jnp.arange(0, n_features, 1).tolist()

        for i in range(n_features):
            for j in range(n_features):

                axs[j, i].grid(True)
                axs[j, i].set_xlim(-1.1, 1.1)
                axs[j, i].set_ylim(-1.1, 1.1)

                reduction_indices = [f_idx for f_idx in feature_indices if not (f_idx == i or f_idx == j)]
                if len(reduction_indices) == n_features - 1:
                    continue

                image = jnp.mean(jnp.abs(abs_map), axis=tuple(reduction_indices))

                if i < j:
                    image = jnp.transpose(image)

                axs[j, i].imshow(image, origin="lower", extent=[-1, 1, -1, 1])
                axs[j, 0].set_ylabel(labels[j])

            axs[-1, i].set_xlabel(labels[i])
        fig.tight_layout()

        return fig, axs

    def visualize_training(self):
        fig, axs = plt.subplots(2, 1, figsize=(6, 4))
        [axs[0].plot(errors) for errors in self.model_errors]

        colors = plt.rcParams["axes.prop_cycle"]()
        c1 = next(colors)["color"]

        mean = jnp.nanmean(jnp.array(self.model_errors), axis=0)
        std = jnp.nanstd(jnp.array(self.model_errors), axis=0)

        axs[1].plot(
            jnp.arange(0, len(mean), 1),
            mean,
            color=c1,
        )
        axs[1].fill_between(
            jnp.arange(0, len(mean), 1),
            mean - std,
            mean + std,
            color=c1,
            alpha=0.1,
        )

        [ax.grid(True) for ax in axs]
        [ax.set_yscale("log") for ax in axs]
        fig.tight_layout()
        return fig, axs

    @property
    def best_model(self) -> eqx.Module:
        """Returns the model with the lowest final prediction error."""
        best_idx = jnp.argmin(self.model_errors[..., -1], axis=0)
        return self.models[best_idx]

    @property
    def median_model(self) -> eqx.Module:
        """Returns the model with the median final prediction error."""
        x = self.model_errors[..., -1].tolist()
        median_idx = np.argpartition(x, len(x) // 2)[len(x) // 2]
        return self.models[median_idx]
