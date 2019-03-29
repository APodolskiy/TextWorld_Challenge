{
training: {
    vocab_size: 20202,
    # actor params
    n_epochs_collection: 1,
    n_played_games: 10,
    n_parallel_envs: 11,
    use_separate_process_envs: true,
    actor_device: "cuda:5",
    target_net_update_freq: 12,
    reward_penalty: 0.3,
    exploration_bonus: 0.35,
    # learner params
    learner_device: "cuda:6",
    chunk_size: 8,
    batch_size: 64,
    lr: 5e-4,
    gamma: 0.9,
    max_steps_per_episode: 50,
    max_samples: 100,
    n_learning_steps: 11,
    saving_freq: 10,
    model_path: "one_skill_model.pth"
    },
network: {
    embedding_dim: 150,
    hidden_size: 512
    },
epsilon: {
    init_eps: 0.0,
    min_eps: 0.01,
    gamma: 0.9,
    step_size: 3
    },
replay_memory : {
    capacity: 500000,
    good_samples_fraction: 0.4,
    bad_samples_fraction: 0.05
    }
}