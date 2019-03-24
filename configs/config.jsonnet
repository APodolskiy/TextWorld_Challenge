{
training: {
    vocab_size: 20202,
    # actor params
    n_epochs_collection: 5,
    n_parallel_envs: 8,
    use_separate_process_envs: true,
    actor_device: "cuda",
    target_net_update_freq: 10,
    # learner params
    learner_device: "cuda",
    batch_size: 64,
    lr: 1e-3,
    gamma: 0.95,
    max_steps_per_episode: 200,
    max_samples: 100,
    n_learning_steps: 20,
    saving_freq: 10,
    model_path: "debug_saved_model.pth"
    },
network: {
    embedding_dim: 150,
    hidden_size: 256
    },
epsilon: {
    init_eps: 0.0,
    min_eps: 0.01,
    gamma: 0.95,
    step_size: 3
    },
replay_memory : {
    capacity: 500000,
    priority_fraction: 0.2
    }
}