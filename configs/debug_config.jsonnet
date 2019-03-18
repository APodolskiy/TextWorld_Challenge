{
training: {
    # actor params
    n_epochs_collection: 5,
    n_parallel_envs: 3,
    use_separate_process_envs: false,
    actor_device: "cuda:5",
    target_net_update_freq: 50,
    # learner params
    learner_device: "cuda:6",
    batch_size: 64,
    lr: 1e-3,
    gamma: 0.95,
    max_steps_per_episode: 200,
    max_samples: 100,
    n_learning_steps: 10,
    saving_freq: 10,
    model_path: "debug_saved_model.pth"
    },
network: {
    embedding_size: 768,
    hidden_size: 30
    },
epsilon: {
    init_eps: 1.0,
    min_eps: 0.05,
    gamma: 0.95,
    step_size: 5
    }
}