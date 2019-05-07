import numpy as np


def generate_session(agent, env, tmax=1000):
    rewards_sequence, actions_probs = [], []
    states, infos = env.reset()

    for t in range(tmax):
        actions, actions_prob = agent.act(states, infos)
        next_states, rewards, dones, infos = env.step(actions)

        actions_probs.append(actions_prob)
        rewards_sequence.append(rewards)

        states = next_states

        if all(dones):
            break

    return np.array(actions_probs).T, np.array(rewards_sequence).T
