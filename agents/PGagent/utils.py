import numpy as np
import textworld
import gym


def generate_session(agent, env, tmax=1000):
    states, actions, rewards = [], [], []
    state, infos = env.reset()

    for t in range(tmax):
        action = agent.act(state, infos)
        next_state, reward, done, infos = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state

        if done:
            break

    return states, actions, rewards