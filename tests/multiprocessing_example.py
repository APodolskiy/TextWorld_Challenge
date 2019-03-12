from collections import namedtuple
from multiprocessing.managers import BaseManager, NamespaceProxy
import os
import random
import torch
import torch.multiprocessing as mp
import time
from typing import NamedTuple


Transition = namedtuple('Transition', ['val1', 'val2'])


class ReplayMemory:
    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, transition: NamedTuple):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        batch_size = min(len(self.buffer), batch_size)
        samples = random.sample(self.buffer, batch_size)
        return samples

    def __len__(self):
        return len(self.buffer)

    def size(self):
        return len(self.buffer)


BaseManager.register('ReplayMemory', ReplayMemory)


class Learner:
    def __init__(self, shared_state, shared_memory):
        self.shared_state = shared_state
        self.shared_memory = shared_memory
        self.batch_size = 10

    def learn(self, t):
        print("Starting learner!")
        while self.shared_memory.size() <= self.batch_size:
            print(f"Learner sleep: {self.shared_memory.size()} {self.batch_size}")
            time.sleep(1)
        for _ in range(t):
            sample = self.shared_memory.sample(self.batch_size)
            print("#" * 30)
            print("Newly sampled data")
            for s in sample:
                print(s[0], s[1])
            print("#" * 30)
            time.sleep(1)


class Actor(mp.Process):
    def __init__(self, shared_state, shared_memory):
        super().__init__()
        self.shared_state = shared_state
        self.shared_memory = shared_memory

    def run(self):
        for _ in range(10):
            self.shared_memory.put([os.getpid(), torch.rand(1)])
            time.sleep(0.5)


def add_experience(shared_mem, replay_mem):
    while True:
        while shared_mem.qsize() or not shared_mem.empty():
            time, value = shared_mem.get()
            replay_mem.push(Transition(time, value))


if __name__ == "__main__":
    mp_manager = mp.Manager()
    shared_state = mp_manager.dict()
    shared_mem = mp_manager.Queue()

    replay_manager = BaseManager()
    replay_manager.start()
    replay_mem = replay_manager.ReplayMemory(100000)

    learner = Learner(shared_state, replay_mem)
    learner_process = mp.Process(target=learner.learn, args=(300,))
    learner_process.start()

    actor_processes = []
    for i in range(10):
        print("Starting actor process")
        actor_process = Actor(shared_state, shared_mem)
        actor_process.start()
        actor_processes.append(actor_process)

    replay_mem_process = mp.Process(target=add_experience, args=(shared_mem, replay_mem))
    replay_mem_process.start()

    learner_process.join()
    [actor_process.join() for actor_process in actor_processes]
    replay_mem_process.join()
