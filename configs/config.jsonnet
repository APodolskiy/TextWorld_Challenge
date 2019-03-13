{
training: {
    n_epochs: 1,
    n_parallel_envs: 2,
    batch_size: 16,
    lr: 1e-3,
    gamma: 0.95,
    max_steps_per_episode: 100,
    actor_device: "cuda"
    },
network: {
    embedding_size: 768,
    hidden_size: 1024
    },
epsilon: {
    init_value: 1.0,
    gamma: 0.95,
    step_size: 10
    }
}