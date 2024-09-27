from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from dmpe.models.model_utils import simulate_ahead


@eqx.filter_grad
def grad_loss(model, true_obs, actions, tau, featurize):

    pred_obs = jax.vmap(simulate_ahead, in_axes=(None, 0, 0, None))(model, true_obs[:, 0, :], actions, tau)

    feat_pred_obs = jax.vmap(featurize, in_axes=(0))(pred_obs)
    feat_true_obs = jax.vmap(featurize, in_axes=(0))(true_obs)

    return jnp.mean((feat_pred_obs - feat_true_obs) ** 2)


@eqx.filter_jit
def make_step(model, observations, actions, tau, opt_state, featurize, optim):
    grads = grad_loss(model, observations, actions, tau, featurize)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state


def dataloader(memory, batch_size, sequence_length, key):
    observations, actions = memory.values()
    observations = jnp.stack(observations, axis=0)
    actions = jnp.stack(actions, axis=0)
    dataset_size = observations.shape[0]

    assert actions.shape[0] == dataset_size - 1

    indices = jnp.arange(dataset_size - sequence_length)

    while True:
        starting_points = jax.random.choice(key=key, a=indices, shape=(batch_size,), replace=True)
        (key,) = jax.random.split(key, 1)

        slice = jnp.linspace(
            start=starting_points, stop=starting_points + sequence_length, num=sequence_length, dtype=int
        ).T

        batched_observations = observations[slice]
        batched_actions = actions[slice]

        yield tuple([batched_observations, batched_actions])


@eqx.filter_jit
def load_single_batch(observations_array, actions_array, starting_points, sequence_length):

    slice = jnp.linspace(
        start=starting_points, stop=starting_points + sequence_length, num=sequence_length, dtype=int
    ).T

    batched_observations = observations_array[slice]
    batched_actions = actions_array[slice]

    batched_observations = batched_observations[:, :, :]
    batched_actions = batched_actions[:, :-1, :]
    return batched_observations, batched_actions


@eqx.filter_jit
def precompute_starting_points(n_train_steps, k, sequence_length, training_batch_size, loader_key):
    index_normalized = jax.random.uniform(loader_key, shape=(n_train_steps, training_batch_size)) * (
        k + 1 - sequence_length
    )
    starting_points = index_normalized.astype(jnp.int32)
    (loader_key,) = jax.random.split(loader_key, 1)

    return starting_points, loader_key


@eqx.filter_jit
def fit(
    model, n_train_steps, starting_points, sequence_length, observations, actions, tau, featurize, optim, init_opt_state
):
    """Fit the model on the gathered data."""

    dynamic_init_model_state, static_model_state = eqx.partition(model, eqx.is_array)
    init_carry = (dynamic_init_model_state, init_opt_state)

    def body_fun(i, carry):
        dynamic_model_state, opt_state = carry
        model_state = eqx.combine(static_model_state, dynamic_model_state)

        batched_observations, batched_actions = load_single_batch(
            observations, actions, starting_points[i, ...], sequence_length
        )
        new_model_state, new_opt_state = make_step(
            model_state, batched_observations, batched_actions, tau, opt_state, featurize, optim
        )

        new_dynamic_model_state, new_static_model_state = eqx.partition(new_model_state, eqx.is_array)
        assert eqx.tree_equal(static_model_state, new_static_model_state) is True
        return (new_dynamic_model_state, new_opt_state)

    final_dynamic_model_state, final_opt_state = jax.lax.fori_loop(
        lower=0, upper=n_train_steps, body_fun=body_fun, init_val=init_carry
    )
    final_model = eqx.combine(static_model_state, final_dynamic_model_state)
    return final_model, final_opt_state


class ModelTrainer(eqx.Module):
    start_learning: jnp.int32
    training_batch_size: jnp.int32
    n_train_steps: jnp.int32
    sequence_length: jnp.int32
    featurize: Callable
    model_optimizer: optax._src.base.GradientTransformationExtraArgs
    tau: jnp.float32

    @eqx.filter_jit
    def fit(self, model, k, observations, actions, opt_state, loader_key):
        starting_points, loader_key = precompute_starting_points(
            n_train_steps=self.n_train_steps,
            k=k,
            sequence_length=self.sequence_length,
            training_batch_size=self.training_batch_size,
            loader_key=loader_key,
        )

        final_model, final_opt_state = fit(
            model=model,
            n_train_steps=self.n_train_steps,
            starting_points=starting_points,
            sequence_length=self.sequence_length,
            observations=observations,
            actions=actions,
            tau=self.tau,
            featurize=self.featurize,
            optim=self.model_optimizer,
            init_opt_state=opt_state,
        )
        return final_model, final_opt_state, loader_key
