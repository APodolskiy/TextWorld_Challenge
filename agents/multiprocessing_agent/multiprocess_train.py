import argparse
import logging
from multiprocessing import Queue
from os.path import exists
from pathlib import Path
from queue import Empty
from shutil import rmtree

import torch.multiprocessing as mp

from agents.multiprocessing_agent.collecting import collect_experience
from agents.multiprocessing_agent.custom_agent import QNet
from agents.multiprocessing_agent.learning import learn
from agents.utils.params import Params
from agents.utils.replay import BinaryPrioritizeReplayMemory

logging.basicConfig(level=logging.INFO)


def clear(q):
    try:
        while True:
            q.get_nowait()
    except Empty:
        pass


if __name__ == "__main__":
    mp.set_start_method("spawn")
    parser = argparse.ArgumentParser(description="Train baseline Q-learning agent.")
    parser.add_argument(
        "games", metavar="game", type=str, help="path to the folder with games"
    )
    args = parser.parse_args()
    train_dir = Path(args.games)
    games = [str(f) for f in train_dir.iterdir() if f.is_file() and f.suffix == ".ulx"][:1]
    print(games)
    params = Params.from_file("configs/config.jsonnet")
    train_params = params.pop("training")

    network_params = params.get("network")
    learner_device = train_params.pop("learner_device")
    policy_net = QNet(network_params, device=learner_device).to(learner_device)
    policy_net.share_memory()

    actor_device = train_params.pop("actor_device")
    target_net = QNet(network_params, device=actor_device).to(actor_device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.share_memory()

    # TODO: change this

    log_dirs = ["runs/actor_run", "runs/learner_run"]
    for log_dir in log_dirs:
        if exists(log_dir):
            rmtree(log_dir)
        Path(log_dir).mkdir(parents=True)
    actor_log_dir, learner_log_dir = log_dirs

    queue = Queue(maxsize=500_000)
    replay = BinaryPrioritizeReplayMemory(capacity=500_000, priority_fraction=0.5)
    processes = []
    collecting_process = mp.Process(
        target=collect_experience,
        kwargs={
            "game_files": games,
            "buffer": queue,
            "train_params": train_params,
            "eps_scheduler_params": params.pop("epsilon"),
            "target_net": target_net,
            "policy_net": policy_net,
            "log_dir": actor_log_dir
        },
    )

    training_process = mp.Process(
        target=learn,
        kwargs={
            "policy_net": policy_net,
            "target_net": target_net,
            "replay_buffer": replay,
            "queue": queue,
            "params": train_params,
            "log_dir": learner_log_dir
        },
    )

    processes.append(collecting_process)
    processes.append(training_process)
    collecting_process.start()
    training_process.start()

    for p in processes:
        p.join()
