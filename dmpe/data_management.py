import pathlib
from enum import Enum
from dataclasses import dataclass


def get_project_root() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent


class ClassicalSystems(Enum):
    FLUID_TANK = 1
    PENDULUM = 2
    CART_POLE = 3


@dataclass(frozen=True)
class DataPaths:
    data_root: pathlib.Path = get_project_root() / "data"
    cs_experiments = get_project_root() / "data" / "classical_systems"
    se_cs_experiments = get_project_root() / "data" / "classical_systems" / "system_excitation_experiments"
    dmpe_cs_experiments = get_project_root() / "data" / "classical_systems" / "system_excitation_experiments" / "dmpe"
    model_learning_experiments = get_project_root() / "data" / "classical_systems" / "model_on_exp_data"
    model_learning_cs_out = (
        get_project_root() / "data" / "classical_systems" / "model_on_exp_data" / "various_exp_together_out"
    )
    pmsm_experiments = get_project_root() / "data" / "pmsm"
    reach_ci_experiments = get_project_root() / "data" / "classical_systems" / "reachable_and_control_invariant_set"
    imperfect_pm_dmpe_experiments = get_project_root() / "data" / "classical_systems" / "imperfect_pm_dmpe"
    sensitivity_analysis_experiments = (
        get_project_root() / "data" / "classical_systems" / "sensitivity_analysis_experiments"
    )
