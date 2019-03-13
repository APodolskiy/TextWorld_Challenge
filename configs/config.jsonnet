{
training: {
    # actor params
    n_epochs_collection: 1000,
    n_parallel_envs: 4,
    use_separate_process_envs: true,
    actor_device: "cuda:6",
    target_net_update_freq: 20,
    # learner params
    learner_device: "cuda:5",
    batch_size: 16,
    lr: 1e-3,
    gamma: 0.95,
    max_steps_per_episode: 100,
    max_samples: 100,
    n_learning_steps: 10000
    },
network: {
    embedding_size: 768,
    hidden_size: 1024
    },
epsilon: {
    init_value: 1.0,
    gamma: 0.95,
    step_size: 500
    }
}