import numpy as np


def generate_session(agent, env, tmax=1000):
    rewards, actions_probs = [], []
    state, infos = env.reset()

    for t in range(tmax):
        actions, actions_prob = agent.act(state, infos)
        next_state, reward, done, infos = env.step(actions)

        actions_probs.append(actions_prob)
        rewards.append(reward)

        state = next_state

        if done:
            break

    return np.array(actions_probs), np.array(rewards)
