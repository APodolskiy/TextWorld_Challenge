from agents.baseline_q_learning.custom_agent import BaseQlearningAgent


def collect_observations(env, agent: BaseQlearningAgent):
    agent.qnet