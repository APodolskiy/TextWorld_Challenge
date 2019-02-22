{
training: {
    n_epochs: 1
    },
agent: {
    device: "cpu",
    max_steps_per_episode: 100,
    network: {
        embedding_size: 768,
        hidden_size: 1024
        }
    }
}