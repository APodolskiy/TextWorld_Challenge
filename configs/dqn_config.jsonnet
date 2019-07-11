{
    training: {
        max_nb_steps_per_episode: 200,  // maximum number of steps that could be taken in each game
        batch_size: 16,  // number of games that are run in parallel
        nb_epochs: 10000, // number of passes through all games
        update_freq: 10,
        target_net_update_freq: 50,
        replay_batch_size: 64,
        clip_grad_norm: 10,
        discount_gamma: 0.95,
    },
    exploration: { // eps-greedy exploration
        init_eps: 1.0,
        final_eps: 0.1,
        steps: 50000,
        print: 50
    },
    model: {
        embedding_size: 100,
        rnn_hidden_size: 256,
        dropout_between_rnn_layers: 0.5
    },
    replay: {
        capacity: 500000,
        priority_fraction: 0.5,
    }
}