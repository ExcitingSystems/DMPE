import optax

import exciting_environments as excenvs


def get_experiment_params(
    setup_name: str,
    env: excenvs.CoreEnvironment,
    featurize: callable,
) -> tuple[dict, dict]:

    if setup_name == "2step":
        model_params = dict(
            obs_dim=env.reset(env.env_properties)[0].shape[0],
            action_dim=env.action_dim,
            width_size=64,
            depth=2,
        )

        lr = 1e-4
        model_trainer_params = dict(
            start_learning=None,
            training_batch_size=128,
            n_train_steps=1_000,
            sequence_length=2,
            featurize=featurize,
            model_optimizer=optax.adabelief(lr),
            tau=env.tau,
        )

    elif setup_name == "10step_small":

        model_params = dict(
            obs_dim=env.reset(env.env_properties)[0].shape[0],
            action_dim=env.action_dim,
            width_size=64,
            depth=2,
        )

        lr = 1e-4
        model_trainer_params = dict(
            start_learning=None,
            training_batch_size=128,
            n_train_steps=1_000,
            sequence_length=10,
            featurize=featurize,
            model_optimizer=optax.adabelief(lr),
            tau=env.tau,
        )

    elif setup_name == "10step_large":

        model_params = dict(
            obs_dim=env.reset(env.env_properties)[0].shape[0],
            action_dim=env.action_dim,
            width_size=128,
            depth=4,
        )

        lr = 1e-4
        model_trainer_params = dict(
            start_learning=None,
            training_batch_size=128,
            n_train_steps=1_000,
            sequence_length=10,
            featurize=featurize,
            model_optimizer=optax.adabelief(lr),
            tau=env.tau,
        )

    elif setup_name == "50step_large":

        model_params = dict(
            obs_dim=env.reset(env.env_properties)[0].shape[0],
            action_dim=env.action_dim,
            width_size=128,
            depth=4,
        )

        lr = 1e-4
        model_trainer_params = dict(
            start_learning=None,
            training_batch_size=128,
            n_train_steps=1_000,
            sequence_length=50,
            featurize=featurize,
            model_optimizer=optax.adabelief(lr),
            tau=env.tau,
        )

    return model_params, model_trainer_params
