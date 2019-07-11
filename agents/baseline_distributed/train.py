import argparse
import json
from multiprocessing.managers import BaseManager
from pathlib import Path

from _jsonnet import evaluate_file
from textworld import EnvInfos
import torch.multiprocessing as mp

from agents.baseline_distributed.actor import Actor
from agents.baseline_distributed.learner import Learner
from agents.utils.replay import TernaryPrioritizeReplayMemory


BaseManager.register('ReplayMemory', TernaryPrioritizeReplayMemory)


def add_experience(shared_mem, replay_mem):
    while 1:
        #print(f"Memory: {shared_mem.qsize()}")
        while shared_mem.qsize() or not shared_mem.empty():
            transition = shared_mem.get()
            replay_mem.push(transition)


def get_additional_infos() -> EnvInfos:
    request_infos = EnvInfos()
    request_infos.description = True
    request_infos.inventory = True
    request_infos.entities, request_infos.verbs = True, True
    request_infos.max_score = True
    request_infos.extras = ["recipe"]
    request_infos.admissible_commands = True
    return request_infos


if __name__ == '__main__':
    mp_manager = mp.Manager()
    shared_state = mp_manager.dict()
    shared_mem = mp_manager.Queue(maxsize=500_000)

    replay_manager = BaseManager()
    replay_manager.start()
    replay_memory = replay_manager.ReplayMemory(100_000)

    config = json.loads(evaluate_file("configs/dqn_config.jsonnet"))

    learner = Learner(config=config, shared_state=shared_state, shared_memory=replay_memory)
    learner_proc = mp.Process(target=learner.learn, args=(5_000_000,))
    learner_proc.start()

    parser = argparse.ArgumentParser(description="Train an agent.")
    parser.add_argument("games", metavar="game", type=str,
                        help="path to the folder with games")
    args = parser.parse_args()
    train_dir = Path(args.games)
    games = [str(f) for f in train_dir.iterdir() if f.is_file() and f.suffix == '.ulx']
    additional_infos = get_additional_infos()

    actor_procs = []
    for i in range(5):
        actor_proc = Actor(actor_id=i,
                           eps=0.2*i,
                           request_infos=additional_infos,
                           game_files=games,
                           config=config,
                           shared_state=shared_state,
                           shared_replay_memory=shared_mem,
                           shared_writer=None)
        actor_proc.start()
        actor_procs.append(actor_proc)

    replay_proc = mp.Process(target=add_experience, args=(shared_mem, replay_memory))
    replay_proc.start()

    learner_proc.join()
    [actor_proc.join() for actor_proc in actor_procs]
    replay_proc.join()
