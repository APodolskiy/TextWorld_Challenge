from multiprocessing import Queue

from agents.multiprocessing_agent.custom_agent import Transition, QNet
from agents.utils.replay import AbstractReplayMemory


def learn(net: QNet, replay_buffer: AbstractReplayMemory, queue: Queue):
    while not queue.empty():
        replay_buffer.push(queue.get())
    batch = replay_buffer.sample(5)
    batch = Transition(*zip(*batch))
    q_values = net(batch.previous_state, batch.action)
    return
