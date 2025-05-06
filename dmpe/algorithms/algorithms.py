from typing import Callable
from tqdm import tqdm

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

import exciting_environments as excenvs
from dmpe.algorithms.algorithm_utils import (
    consult_exciter,
    interact_and_observe,
    default_dmpe_parameterization,
)
from dmpe.excitation.excitation_utils import loss_function, Exciter
from dmpe.models.model_training import ModelTrainer
from dmpe.utils.density_estimation import (
    DensityEstimate,
    build_grid,
)
from dmpe.utils.metrics import JSDLoss


def excite_and_fit(
    n_time_steps: int,
    env: excenvs.CoreEnvironment,
    model: eqx.Module,
    obs: jax.Array,
    state: excenvs.CoreEnvironment.State,
    proposed_actions: jax.Array,
    exciter: Exciter,
    model_trainer: ModelTrainer,
    density_estimate: DensityEstimate,
    observations: jax.Array,
    actions: jax.Array,
    opt_state_model: optax.OptState,
    loader_key: jax.random.PRNGKey,
    expl_key: jax.random.PRNGKey,
    callback_every: int,
    callback: Callable | None = None,
) -> tuple[jax.Array, jax.Array, eqx.Module, DensityEstimate]:
    """
    Main algorithm to apply to a given (unknown) system and generate informative data from that system.

    A pseudocode description of this algorithm is given in the corresponding publication [Vater2024].
    In summary, the algorithm iterates over the time steps k. Each iteration an action is chosen to be
    applied by the Exciter object. Afterwards, the action is applied to the system and its effect is
    observed. This is followed up by the (optional) update to the dynamics model.
    The rest of the code is only for monitoring and, finally, progressing to the next time step k+1.

    Args:
        n_time_steps (int): The number of time steps to run the algorithm for.
        env (excenvs.CoreEnvironment): The environment object representing the system.
        model (eqx.Module): The model used for prediction.
        obs (jax.Array): The initial observation of the system.
        state (excenvs.CoreEnvironment.State): The initial state of the system.
        proposed_actions (jax.Array): The proposed actions for exploration.
        exciter (Exciter): The exciter object responsible for choosing actions.
        model_trainer (ModelTrainer): The model trainer object responsible for training the model.
        density_estimate (DensityEstimate): The density estimate used for exploration.
        observations (jax.Array): The history of observations.
        actions (jax.Array): The history of actions.
        opt_state_model (optax.OptState): The optimizer state for the model.
        loader_key (jax.random.PRNGKey): The key used for loading data.
        expl_key: (jax.random.PRNGKey): The key used for random action generation.
        callback_every (int): The frequency at which to run the callback function.
        callback (Callable | None): Implementation of the callback function.

    Returns:
        tuple[jax.Array, jax.Array, eqx.Module, DensityEstimate, list, jax.Array, list]: A tuple containing
        the history of observations, the history of actions, the updated model, the updated density estimate,
        the prediction losses, the proposed actions, and the callback output.
    """
    prediction_losses = []
    data_losses = []

    callback_out = []

    for k in tqdm(range(n_time_steps)):
        action, next_proposed_actions, next_density_estimate, prediction_loss, next_expl_key = consult_exciter(
            k=k,
            exciter=exciter,
            obs=obs,
            state=state,
            model=model,
            density_estimate=density_estimate,
            proposed_actions=proposed_actions,
            expl_key=expl_key,
        )

        prediction_losses.append(prediction_loss)  # predicted loss for the last excitation optimization

        next_obs, next_state, actions, observations = interact_and_observe(
            env=env, k=jnp.array([k]), action=action, state=state, actions=actions, observations=observations
        )

        if model_trainer is not None:
            if k > model_trainer.start_learning:
                model, opt_state_model, loader_key = model_trainer.fit(
                    model=model,
                    k=jnp.array([k]),
                    observations=observations,
                    actions=actions,
                    opt_state=opt_state_model,
                    loader_key=loader_key,
                )
        elif hasattr(model, "fit"):
            model = model.fit(model, jnp.array([k]), observations, actions)
        else:
            if k == 0:
                print("Model is used statically and not re-fitted or updated otherwise.")

        ## Start Monitoring
        # evaluate the current excitation metric value for the acquired data
        # (Only necessary for monitoring purposes)
        data_loss = JSDLoss(
            next_density_estimate.p / jnp.sum(next_density_estimate.p),
            exciter.target_distribution / jnp.sum(exciter.target_distribution),
        )
        data_losses.append(data_loss)

        # callback
        if k % callback_every == 0 and k > 0:
            if callback is not None:
                callback_out.append(
                    callback(
                        k=jnp.array([k]),
                        env=env,
                        obs=obs,
                        state=state,
                        action=action,
                        next_obs=next_obs,
                        next_state=next_state,
                        observations=observations,
                        actions=actions,
                        model=model,
                        density_estimate=density_estimate,
                        proposed_actions=proposed_actions,
                        next_density_estimate=next_density_estimate,
                        next_proposed_actions=next_proposed_actions,
                        data_losses=data_losses,
                        prediction_losses=prediction_losses,
                    )
                )
        ## End Monitoring

        # k <- k + 1
        obs = next_obs
        state = next_state
        proposed_actions = next_proposed_actions
        density_estimate = next_density_estimate
        expl_key = next_expl_key

    return observations, actions, model, density_estimate, prediction_losses, proposed_actions, callback_out


def excite_with_dmpe(
    env: excenvs.CoreEnvironment,
    exp_params: dict,
    proposed_actions: jax.Array,
    loader_key: jax.random.PRNGKey,
    expl_key: jax.random.PRNGKey,
    callback_every: int | None = None,
    callback: Callable | None = None,
):
    """
    Excite the system using the Differentiable Model Predictive Excitation (DMPE) algorithm.

    Args:
        env (excenvs.CoreEnvironment): The environment object representing the system.
        exp_params (dict): The experiment parameters.
        proposed_actions (jax.Array): The initial proposed actions to apply.
        loader_key (jax.random.PRNGKey): The key used for loading data.
        expl_key (jax.random.PRNGKey): The key used for random action generation.
        callback_every (int | None): The frequency of steps at which to run the callback function.
            If it is 'None' no callback is done.
        callback (Callable): Callback function to monitor the excitation process.
            See 'dmpe/evaluation/callbacks.py' for examples and API.

    Returns:
        Tuple[jax.Array, jax.Array, eqx.Module, DensityEstimate, list, jax.Array, list]: A tuple containing
        the history of observations, the history of actions, the updated model, the updated density estimate,
        the prediction losses, the proposed actions, and the callback output.
    """
    obs, state = env.reset(env.env_properties)

    dim_obs_space = obs.shape[0]
    dim_action_space = env.action_dim

    if exp_params["alg_params"]["consider_action_distribution"]:
        dim = dim_obs_space + dim_action_space
    else:
        dim = dim_obs_space

    n_grid_points = exp_params["alg_params"]["points_per_dim"] ** dim

    # setup memory variables
    observations = jnp.zeros((exp_params["n_time_steps"], dim_obs_space))
    observations = observations.at[0].set(obs)
    actions = jnp.zeros((exp_params["n_time_steps"] - 1, dim_action_space))

    exciter = Exciter(
        start_optimizing=exp_params["alg_params"]["start_optimizing"],
        loss_function=loss_function,
        grad_loss_function=jax.value_and_grad(loss_function, argnums=(3)),
        excitation_optimizer=exp_params["alg_params"]["excitation_optimizer"],
        tau=env.tau,
        n_opt_steps=exp_params["alg_params"]["n_opt_steps"],
        consider_action_distribution=exp_params["alg_params"]["consider_action_distribution"],
        target_distribution=exp_params["alg_params"]["target_distribution"],
        penalty_function=exp_params["alg_params"]["penalty_function"],
        clip_action=exp_params["alg_params"]["clip_action"],
        n_starts=exp_params["alg_params"]["n_starts"],
        reuse_proposed_actions=exp_params["alg_params"]["reuse_proposed_actions"],
    )

    if exp_params["model_trainer_params"] is None and exp_params["model_params"] is None:
        model_trainer = None
        model = env
        opt_state_model = None
    elif exp_params["model_trainer_params"] is None and exp_params["model_params"] is not None:
        model_trainer = None
        model = exp_params["model_class"](**exp_params["model_params"])
        opt_state_model = None
    else:
        model_trainer = ModelTrainer(
            start_learning=exp_params["model_trainer_params"]["start_learning"],
            training_batch_size=exp_params["model_trainer_params"]["training_batch_size"],
            n_train_steps=exp_params["model_trainer_params"]["n_train_steps"],
            sequence_length=exp_params["model_trainer_params"]["sequence_length"],
            featurize=exp_params["model_trainer_params"]["featurize"],
            model_optimizer=optax.adabelief(exp_params["model_trainer_params"]["model_lr"]),
            tau=env.tau,
        )
        model = exp_params["model_class"](**exp_params["model_params"])
        opt_state_model = model_trainer.model_optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    density_estimate = DensityEstimate(
        p=jnp.zeros([n_grid_points, 1]),
        z_g=build_grid(
            dim,
            low=-exp_params["alg_params"]["grid_extend"],
            high=exp_params["alg_params"]["grid_extend"],
            points_per_dim=exp_params["alg_params"]["points_per_dim"],
        ),
        bandwidth=jnp.array([exp_params["alg_params"]["bandwidth"]]),
        n_observations=jnp.array([0]),
    )

    observations, actions, model, density_estimate, losses, proposed_actions, callback_out = excite_and_fit(
        n_time_steps=exp_params["n_time_steps"],
        env=env,
        model=model,
        obs=obs,
        state=state,
        proposed_actions=proposed_actions,
        exciter=exciter,
        model_trainer=model_trainer,
        density_estimate=density_estimate,
        observations=observations,
        actions=actions,
        opt_state_model=opt_state_model,
        loader_key=loader_key,
        expl_key=expl_key,
        callback_every=callback_every if callback_every is not None else exp_params["n_time_steps"] + 1,
        callback=callback,
    )

    return observations, actions, model, density_estimate, losses, proposed_actions, callback_out


def default_dmpe(env, seed=0, n_time_steps=5000, featurize=None, model_class=None, callback=None, callback_every=None):
    """Runs DMPE with default parameterization. The parameter choices might
    not be optimal for a given system.

    In future work, automatic tuning for the parameters will be added such that no
    manual tuning is required.

    Args:
        env: The environment object representing the system.
        seed (int): The random seed for reproducibility.
        n_time_steps (int): The number of time steps to run the algorithm for.
        featurize: The function used for feature applied onto the observations.
        model_class: The class of the model used for prediction.
        callback (Callable): Callback function to monitor the excitation process.
            See 'dmpe/evaluation/callbacks.py' for examples and API.
        callback_every (int | None): The frequency of steps at which to run the callback function.
            If it is 'None' no callback is done.

    Returns:
        Tuple[jax.Array, jax.Array, eqx.Module, DensityEstimate, list, jax.Array, list]: A tuple containing
        the history of observations, the history of actions, the updated model, the updated density estimate,
        the prediction losses, the proposed actions, and the callback output.
    """

    return excite_with_dmpe(
        env,
        *default_dmpe_parameterization(env, seed, n_time_steps, featurize, model_class),
        callback_every=callback_every,
        callback=callback,
    )
