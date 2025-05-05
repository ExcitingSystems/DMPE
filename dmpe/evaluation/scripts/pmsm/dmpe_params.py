from typing import Callable
import optax

from dmpe.models.models import NeuralEulerODEPMSM
from dmpe.models.rls import SimulationPMSM_RLS
from dmpe.utils.density_estimation import get_uniform_target_distribution


def get_alg_params(consider_action_distribution: bool, penalty_function: Callable):
    """Get parameters for the DMPE algorithm in the PMSM experiments."""

    alg_params = dict(
        bandwidth=0.08,
        n_prediction_steps=5,
        points_per_dim=21,
        grid_extend=1.05,
        excitation_optimizer=optax.adabelief(1e-2),
        n_opt_steps=200,
        start_optimizing=5,
        consider_action_distribution=consider_action_distribution,
        penalty_function=penalty_function,
        target_distribution=None,
        clip_action=False,
        n_starts=10,
        reuse_proposed_actions=True,
    )

    alg_params["target_distribution"] = get_uniform_target_distribution(
        dim=4 if consider_action_distribution else 2,
        points_per_dim=alg_params["points_per_dim"],
        bandwidth=alg_params["bandwidth"],
        grid_extend=alg_params["grid_extend"],
        consider_action_distribution=consider_action_distribution,
        penalty_function=penalty_function,
    )
    return alg_params


def get_RLS_params(consider_action_distribution, penalty_function):
    """Get parameters for the application of RLS models for the DMPE pmsm experiments."""
    alg_params = get_alg_params(
        consider_action_distribution=consider_action_distribution, penalty_function=penalty_function
    )

    model_params = dict(lambda_=0.9)
    model_trainer_params = None
    model_class = SimulationPMSM_RLS

    return alg_params, model_params, model_class, model_trainer_params


def get_NODE_params(consider_action_distribution, penalty_function):
    """Get parameters for the application of NODE models for the DMPE pmsm experiments."""

    alg_params = get_alg_params(
        consider_action_distribution=consider_action_distribution, penalty_function=penalty_function
    )

    model_params = dict(obs_dim=2, action_dim=2, width_size=64, depth=3, key=None)
    model_trainer_params = dict(
        start_learning=alg_params["n_prediction_steps"],
        training_batch_size=64,
        n_train_steps=5,
        sequence_length=alg_params["n_prediction_steps"],
        featurize=lambda x: x,
        model_lr=1e-4,
    )
    model_class = NeuralEulerODEPMSM

    return alg_params, model_params, model_class, model_trainer_params


def get_PM_params(consider_action_distribution, penalty_function):
    alg_params = get_alg_params(
        consider_action_distribution=consider_action_distribution, penalty_function=penalty_function
    )

    alg_params["n_opt_steps"] = 100

    model_params = None
    model_trainer_params = None
    model_class = None

    return alg_params, model_params, model_class, model_trainer_params
