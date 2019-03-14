{
training: {
    # actor params
    n_epochs_collection: 100000,
    n_parallel_envs: 4,
    use_separate_process_envs: true,
    actor_device: "cuda:6",
    target_net_update_freq: 20,
    # learner params
    learner_device: "cuda:5",
    batch_size: 10,
    lr: 5e-4,
    gamma: 0.95,
    max_steps_per_episode: 100,
    max_samples: 100,
    n_learning_steps: 10000000,
    saving_freq: 10,
    model_path: "saved_model.pth"
    },
network: {
    embedding_size: 768,
    hidden_size: 800
    },
epsilon: {
    init_eps: 1.0,
    min_eps: 0.01,
    gamma: 0.92,
    step_size: 5
    }
}