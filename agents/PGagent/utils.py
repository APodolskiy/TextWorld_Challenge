import numpy as np


def generate_session(agent, env, tmax=1000):
    rewards, action_probs = [], []
    state, infos = env.reset()

    for t in range(tmax):
        action, action_prob = agent.act(state, infos)
        next_state, reward, done, infos = env.step(action)

        action_probs.append(action_prob)
        rewards.append(reward)

        state = next_state

        if done:
            break

    return np.array(action_probs), np.array(rewards)
