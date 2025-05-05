import jax
import argparse
import eval_dmpe
import eval_goats


rpms = [0, 3000, 5000, 7000, 9000]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DMPE on the PMSM environment.")
    parser.add_argument("--DMPE", action=argparse.BooleanOptionalAction)
    parser.add_argument("--iGOATS", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    if args.DMPE:
        model_names = ["RLS", "NODE", "PM"]
        consider_actions = [True]

        for consider_action in consider_actions:
            for model_name in model_names:
                for rpm in rpms:
                    eval_dmpe.main(rpm, model_name, consider_action)

    elif args.iGOATS:
        jax.config.update("jax_platform_name", "cpu")

        consider_action = True
        for rpm in rpms:
            eval_goats.main(rpm, consider_action)

    else:
        raise ValueError("Please specify either DMPE or iGOATS")
