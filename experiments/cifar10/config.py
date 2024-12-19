import ml_collections


def new_dict(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.rng_key = 1

    config.nn = new_dict(
        dlatent=128,
        dhid=64,
    )
    config.data = new_dict(
        dataset="cifar10",
        shape=(32, 32, 3),
        n_classes=10,
        n_out_channels=3,
    )
    config.training = new_dict(
        n_steps=500_000,
        batch_size=512,
        buffer_size=512 * 10,
        prefetch_size=512 * 2,
        do_reshuffle=True,
        checkpoints=new_dict(
            max_to_keep=10,
            save_interval_steps=1,
        ),
        n_eval_frequency=10_000,
        n_checkpoint_frequency=30_000,
        n_eval_batches=10,
        n_sampling_frequency=50_000,
    )

    config.optimizer = new_dict(
        name="adam",
        params=new_dict(
            learning_rate=0.0002,
            weight_decay=1e-6,
            do_warmup=True,
            warmup_steps=1_000,
            do_decay=True,
            decay_steps=500_000,
            end_learning_rate=1e-6,
            init_learning_rate=1e-8,
            do_gradient_clipping=True,
            gradient_clipping=1.0,
        ),
    )

    return config
