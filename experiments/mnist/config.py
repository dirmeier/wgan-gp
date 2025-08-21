import ml_collections


def new_dict(**kwargs):
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
  config = ml_collections.ConfigDict()
  config.rng_key = 1

  batchsize = 512 * 4
  config.nn = new_dict(
    dlatent=64,
    dchan=512,
    base_resolution=(4, 4),
  )
  config.data = new_dict(
    dataset="mnist",
    shape=(32, 32, 1),
  )
  config.training = new_dict(
    n_steps=100_000,
    n_update_generator=5,
    batch_size=batchsize,
    buffer_size=batchsize * 10,
    prefetch_size=batchsize * 2,
    do_reshuffle=True,
    checkpoints=new_dict(
      max_to_keep=10,
      save_interval_steps=1,
    ),
    n_eval_frequency=5_000,
    n_eval_batches=20,
    n_sampling_frequency=20_000,
  )

  config.optimizer = new_dict(
    name="adam",
    params=new_dict(
      learning_rate=0.0002,
      b1=0.5,
      b2=0.999,
      weight_decay=1e-6,
      do_warmup=True,
      warmup_steps=1_000,
      do_decay=True,
      decay_steps=100_000,
      end_learning_rate=1e-6,
      init_learning_rate=1e-8,
      do_gradient_clipping=True,
      gradient_clipping=1.0,
    ),
  )

  return config
