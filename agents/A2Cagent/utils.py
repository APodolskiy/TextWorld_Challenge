from collections import deque


def generate_session(agent, env, tmax=1000):
    rewards_sequence, actions_probs = [], []
    states, infos = env.reset()
    states_sequence = [states]
    next_states_sequence = []

    for t in range(tmax):
        actions, actions_prob = agent.act(states, next_states, infos)
        next_states, rewards, dones, infos = env.step(actions)

        actions_probs.append(actions_prob)
        rewards_sequence.append(rewards)

        states_sequence.append(states)
        next_states_sequence.append(next_states)
        states = next_states

        if all(dones):
            break

    return states_sequence, next_states_sequence, actions_probs, \
           rewards_sequence, sum(infos["has_won"]), sum(infos["has_lost"])
