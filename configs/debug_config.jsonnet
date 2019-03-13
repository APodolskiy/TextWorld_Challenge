{
training: {
    # actor params
    n_epochs_collection: 1,
    n_parallel_envs: 2,
    use_separate_process_envs: false,
    actor_device: "cuda",
    target_net_update_freq: 100,
    # learner params
    learner_device: "cuda",
    batch_size: 16,
    lr: 1e-3,
    gamma: 0.95,
    max_steps_per_episode: 100,
    max_samples: 100,
    n_learning_steps: 1000
    },
network: {
    embedding_size: 768,
    hidden_size: 1024
    },
epsilon: {
    init_value: 0.9,
    gamma: 0.95,
    step_size: 100
    }
}