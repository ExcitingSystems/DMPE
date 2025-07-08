import pathlib
from dataclasses import dataclass
import equinox as eqx


def get_project_root() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent


@dataclass
class DataPaths:
    data_root: pathlib.Path = get_project_root() / "data"
    cs_experiments = get_project_root() / "data" / "classical_systems"
    dmpe_cs_experiments = get_project_root() / "data" / "classical_systems" / "dmpe"
    pmsm_experiments = get_project_root() / "data" / "pmsm"
