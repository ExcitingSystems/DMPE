from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from dmpe.models.model_utils import simulate_ahead


@eqx.filter_jit
@eqx.filter_grad
def grad_loss(
    model: eqx.Module,
    true_obs: jax.Array,
    actions: jax.Array,
    tau: float,
    featurize: Callable,
) -> jax.Array:
    """Basic MSE implementation for model training.

    The model predicts the observation sequence based on the actions to be applied
    and the first element in the sequence. The predicted observations are then
    compared with the ground truth values.

    Note: The function itself actually returns the MSE value. Due to the gradient
    decorator the function returns the gradient of the MSE with respect to the
    model parameters instead. This is also why the function is called grad_loss
    and not loss.

    Args:
        model (eqx.Module): The model to be trained.
        true_obs (jax.Array): A batched sequence of ground-truth observations with
            shape (batch_size, sequence_length, obs_dim)
        actions (jax.Array): The actions that have been applied to generate the observation
            sequence.
        tau (float): Sampling frequency of the system.
        featurize (Callable): Function that potentially adds additional features to the
            observations. For instance angle information is usually transformed to sin
            and cos of the angle to have a better grasp on similar angles.

    Returns:
        The resulting MSE loss value (jax.Array)
        (Due to the gradient decorator this actually returns the gradient of the MSE w.r.t. the model parameters)
    """
    pred_obs = jax.vmap(simulate_ahead, in_axes=(None, 0, 0, None))(model, true_obs[:, 0, :], actions, tau)

    feat_pred_obs = jax.vmap(featurize, in_axes=(0))(pred_obs)
    feat_true_obs = jax.vmap(featurize, in_axes=(0))(true_obs)

    return jnp.mean((feat_pred_obs - feat_true_obs) ** 2)


@eqx.filter_jit
def make_step(
    model: eqx.Module,
    observations: jax.Array,
    actions: jax.Array,
    tau: float,
    opt_state: optax.OptState,
    featurize: Callable,
    optim: optax.GradientTransformation,
) -> tuple[eqx.Module, optax.OptState]:
    """Performs a single update step on the model.

    First the gradient of the MSE loss with respect to the model parameters is computed.
    Then the optimizers tunes the model parameters based on this.

    Args:
        model (eqx.Module): The model to be trained.
        observations (jax.Array): A batched sequence of ground-truth observations with
            shape (batch_size, sequence_length, obs_dim)
        actions (jax.Array): The actions that have been applied to generate the observation
            sequence.
        tau (float): Sampling frequency of the system.
        opt_state (optax.OptState): State of the model optimizer
        featurize (Callable): Function that potentially adds additional features to the
            observations. For instance angle information is usually transformed to sin
            and cos of the angle to have a better grasp on similar angles.
        optim (optax.GradientTransformation): Model optimizer

    Returns:
        model (eqx.Module): The updated model
        opt_state (optax.OptState): The updated state of the model optimizer
    """
    grads = grad_loss(model, observations, actions, tau, featurize)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state


@eqx.filter_jit
def load_single_batch(
    observations_array: jax.Array,
    actions_array: jax.Array,
    starting_points: jax.Array,
    sequence_length: int,
) -> tuple[jax.Array, jax.Array]:
    """Loads a single batch of data from the memory arrays.

    Args:
        observations_array (jax.Array): An array containing all gathered observations. Note
            that the array is zero-padded to always have the same length. The shape is always
            (max_length, obs_dim) regardless of how many observations < max_length have been
            gathered.
        actions_array (jax.Array): Array with actions corresponding to the observations array.
        starting_points (jax.Array): The starting points for all batches with shape (batch_size,)
        sequence_length (int): The length of the sequences in each batch. The minimum sensible
            sequence length is 2. As the actions array is empty otherwise and there is only
            a single observation, which one cannot properly learn with.

    Returns:
        batched_observations (jax.Array): A batch of observations with shape (batch_size, sequence_length, obs_dim)
        batched_actions (jax.Array): A batch of actions with shape (batch_size, sequence_length-1, action_dim)
    """

    slice = jnp.linspace(
        start=starting_points, stop=starting_points + sequence_length, num=sequence_length, dtype=int
    ).T

    batched_observations = observations_array[slice]
    batched_actions = actions_array[slice]

    batched_observations = batched_observations[:, :, :]
    batched_actions = batched_actions[:, :-1, :]
    return batched_observations, batched_actions


@eqx.filter_jit
def precompute_starting_points(
    n_train_steps: int,
    k: jax.Array,
    sequence_length: int,
    training_batch_size: int,
    loader_key: jax.random.PRNGKey,
) -> tuple[jax.Array, jax.random.PRNGKey]:
    """Sample where the batch sequences should start in the gathered data arrays.

    The maximum possible starting index is k + 1 - sequence_length as beyond that
    the data arrays are filled with zeros.

    Args:
        n_train_steps (int): Number of training steps to be performed.
        k (jax.Array): Denotes to which index the memory arrays are currently filled.
        sequence_length (int): The length of the sequences in each batch. The minimum sensible
            sequence length is 2. As the actions array is empty otherwise and there is only
            a single observation, which one cannot properly learn with.
        training_batch_size (int): The size of the training batches.
        loader_key (jax.random.PRNGKey): Random key for sampling.

    Returns:
        starting_points (jax.Array): The starting points for all batches with shape (n_train_steps, training_batch_size)
        next_loader_key (jax.random.PRNGKey): The updated random key.
    """
    next_loader_key, momentary_loader_key = jax.random.split(loader_key, 2)

    index_normalized = jax.random.uniform(momentary_loader_key, shape=(n_train_steps, training_batch_size)) * (
        k + 1 - sequence_length
    )
    starting_points = index_normalized.astype(jnp.int32)

    return starting_points, next_loader_key


@eqx.filter_jit
def fit(
    model: eqx.Module,
    n_train_steps: int,
    starting_points: jax.Array,
    sequence_length: int,
    observations: jax.Array,
    actions: jax.Array,
    tau: float,
    featurize: Callable,
    optim: optax.GradientTransformation,
    init_opt_state: optax.OptState,
) -> tuple[eqx.Module, optax.OptState]:
    """Fit the model on the gathered data.

    For 'n_train_steps' iterations, a batch of observations and actions is drawn
    according to the corresponding line in the starting_points array. In each iteration
    a training step is performed based on the drawn batches of data.

    Args:
        model (eqx.Module): The model to be trained.
        n_train_steps (int): Number of consecutive training steps to performed.
        starting_points (jax.Array): The starting points for all batches with shape
            (n_train_steps, training_batch_size)
        sequence_length (int): The length of the sequences in each batch. The minimum sensible
            sequence length is 2. As the actions array is empty otherwise and there is only
            a single observation, which one cannot properly learn with.
        observations (jax.Array): An array containing all gathered observations. Note
            that the array is zero-padded to always have the same length. The shape is always
            (max_length, obs_dim) regardless of how many observations < max_length have been
            gathered.
        actions (jax.Array): Array with actions corresponding to the observations array.
        tau (float): Sampling frequency of the system.
        featurize (Callable): Function that potentially adds additional features to the
            observations. For instance angle information is usually transformed to sin
            and cos of the angle to have a better grasp on similar angles.
        optim (optax.GradientTransformation): Model optimizer
        init_opt_state (optax.OptState): State of the model optimizer

    Returns:
        model (eqx.Module): The updated model
        opt_state (optax.OptState): The updated state of the model optimizer
    """

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
    """A class that carries the necessary tools for training simulation models based on eqx.Modules.

    Args:
        start_learning (int): How many steps the trainer should wait before starting the training.
            This is done so that a minimum of data can be gathered before training is started.
        training_batch_size (int): The size of the training batches.
        n_train_steps (int): Number of consecutive training steps to performed.
        sequence_length (int): The length of the sequences in each batch. The minimum sensible
            sequence length is 2. As the actions array is empty otherwise and there is only
            a single observation, which one cannot properly learn with.
        featurize (Callable): Function that potentially adds additional features to the observations.
        model_optimizer (optax.GradientTransformation): Model optimizer
        tau (float): Sampling frequency of the system.

    """

    start_learning: int
    training_batch_size: int
    n_train_steps: int
    sequence_length: int
    featurize: Callable
    model_optimizer: optax.GradientTransformation
    tau: float

    @eqx.filter_jit
    def fit(
        self,
        model: eqx.Module,
        k: jax.Array,
        observations: jax.Array,
        actions: jax.Array,
        opt_state: optax.OptState,
        loader_key: jax.random.PRNGKey,
    ) -> tuple[eqx.Module, optax.OptState, jax.random.PRNGKey]:
        """Sample batches and train the model based on the data gathered up to this point.

        Args:
            model (eqx.Module): The model to be trained.
            k (jax.Array): Denotes to which index the memory arrays are currently filled.
            observations (jax.Array): An array containing all gathered observations. Note
                that the array is zero-padded to always have the same length. The shape is always
                (max_length, obs_dim) regardless of how many observations < max_length have been
                gathered.
            actions (jax.Array): Array with actions corresponding to the observations array.
            opt_state (optax.OptState): State of the model optimizer
            loader_key (jax.random.PRNGKey): Random key for sampling.
        Returns:
            final_model (eqx.Module): The updated model
            final_opt_state (optax.OptState): The updated state of the model optimizer
            next_loader_key (jax.random.PRNGKey): The updated random key.
        """
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
