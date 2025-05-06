import argparse
import pathlib

REPO_ROOT_PATH = pathlib.Path(__file__).parent.parent.parent.parent
DATA_PATH = REPO_ROOT_PATH / pathlib.Path("data")

if __name__ == "__main__":
    print("repo root: ", REPO_ROOT_PATH)

    print(f"Creating 'data' folder at '{DATA_PATH}'")
    DATA_PATH.mkdir(parents=False, exist_ok=True)

    parser = argparse.ArgumentParser(description="Create experiments folder structure.")
    parser.add_argument("--which", type=str, default="all", help="One of 'all', 'classical', or 'pmsm'.")

    args = parser.parse_args()
    which = args.which

    if which == "classical" or which == "all":

        systems = ["fluid_tank", "pendulum", "cart_pole"]
        algorithms = ["dmpe", "igoats", "sgoats", "perfect_model_dmpe"]

        for algo in algorithms:
            for system in systems:
                system_algo_path = (
                    DATA_PATH / pathlib.Path("classical_systems") / pathlib.Path(algo) / pathlib.Path(system)
                )
                system_algo_path.mkdir(parents=True, exist_ok=True)

    if which == "pmsm" or which == "all":
        structure = {
            "dmpe": ["NODE", "PM", "RLS"],
            "igoats": ["_"],
            "heuristics": ["current_plane_sweep", "random_walk"],
        }

        for algo_class, algo_names in structure.items():
            for algo_name in algo_names:
                algo_path = DATA_PATH / pathlib.Path("pmsm") / pathlib.Path(algo_class) / pathlib.Path(algo_name)
                algo_path.mkdir(parents=True, exist_ok=True)

    print("Successfully created all directories.")
