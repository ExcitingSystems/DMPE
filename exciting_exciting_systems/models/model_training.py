import jax
import jax.numpy as jnp
import equinox as eqx

from exciting_exciting_systems.models.model_utils import simulate_ahead


@eqx.filter_value_and_grad
def grad_loss(model, true_obs, actions, tau, featurize):

    pred_obs = jax.vmap(simulate_ahead, in_axes=(None, 0, 0, None))(model, true_obs[:, 0, :], actions, tau)

    feat_pred_obs = jax.vmap(featurize, in_axes=(0))(pred_obs)
    feat_true_obs = jax.vmap(featurize, in_axes=(0))(true_obs)

    return jnp.mean((feat_pred_obs - feat_true_obs) ** 2)


@eqx.filter_jit
def make_step(model, observations, actions, tau, opt_state, featurize, optim):
    loss, grads = grad_loss(model, observations, actions, tau, featurize)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


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
            start=starting_points,
            stop=starting_points+sequence_length,
            num=sequence_length,
            dtype=int
        ).T

        batched_observations = observations[slice]
        batched_actions = actions[slice]

        yield tuple([batched_observations, batched_actions])


@eqx.filter_jit
def load_single_batch(observations_array, actions_array, starting_points, sequence_length):

    slice = jnp.linspace(
            start=starting_points,
            stop=starting_points+sequence_length,
            num=sequence_length,
            dtype=int
        ).T

    batched_observations = observations_array[slice]
    batched_actions = actions_array[slice]

    batched_observations = batched_observations[:, :, :]
    batched_actions = batched_actions[:, :-1, :]
    return batched_observations, batched_actions
