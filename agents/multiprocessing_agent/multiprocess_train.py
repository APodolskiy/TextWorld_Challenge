import argparse
import logging
from multiprocessing import Queue

from pathlib import Path
from queue import Empty
from time import sleep

import gym
import textworld
from tqdm import tqdm
import torch.multiprocessing as mp
from agents.multiprocessing_agent.advanced_agent import QNet
from agents.multiprocessing_agent.custom_agent import BaseQlearningAgent
from agents.utils.params import Params
from agents.utils.replay import ExperienceReplay
from test_submission import _validate_requested_infos


def clear(q):
    try:
        while True:
            q.get_nowait()
    except Empty:
        pass


def collect_experience(game_files, main_net, target_net, buffer: Queue, params):

    # TODO: get from main_net

    logging.basicConfig(level=logging.INFO)
    train_params = params.pop("training")

    actor = BaseQlearningAgent(
        net=target_net, experience_replay_buffer=buffer, config=params.pop("agent")
    )

    requested_infos = actor.select_additional_infos()
    _validate_requested_infos(requested_infos)

    env_id = textworld.gym.register_games(
        game_files,
        requested_infos,
        max_episode_steps=actor.max_steps_per_episode,
        name="training",
    )
    batch_size = 2
    env_id = textworld.gym.make_batch(env_id, batch_size=batch_size, parallel=True)
    env = gym.make(env_id)

    for epoch_no in range(1, train_params.pop("n_epochs") + 1):
        stats = {"scores": [], "steps": []}
        for _ in tqdm(range(len(game_files))):
            obs, infos = env.reset()
            actor.train()

            dones = [False for _ in range(batch_size)]
            steps = [0 for _ in range(batch_size)]
            scores = [0 for _ in range(batch_size)]
            while not all(dones):
                # Increase step counts.
                steps = [step + int(not done) for step, done in zip(steps, dones)]
                command = actor.act(obs, scores, dones, infos)
                obs, scores, dones, infos = env.step(command)

        score = sum(stats["scores"])
        steps = sum(stats["steps"])
        print(f"Epoch: {epoch_no:3d} | {score:2.1f} pts | {steps:4.1f} steps")
    print("Returning")
    return


def train_net(net, queue, buffer: ExperienceReplay):

    # TODO: actually train

    for _ in range(10):
        sleep(1)
        for _ in range(4):
            sample = queue.get()
            buffer.push(sample)

        try:
            batch = buffer.sample(10)
            print(batch)
        except ValueError as e:
            print(e)
    clear(queue)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline Q-learning agent.")
    parser.add_argument(
        "games", metavar="game", type=str, help="path to the folder with games"
    )
    args = parser.parse_args()
    train_dir = Path(args.games)
    games = [str(f) for f in train_dir.iterdir() if f.is_file() and f.suffix == ".ulx"]
    params = Params.from_file("configs/config.jsonnet")
    agent_params = params.get("agent")
    target_net = QNet(agent_params.get("network"))
    target_net.share_memory()
    main_net = QNet(agent_params.get("network"))
    main_net.share_memory()

    processes = []
    queue = Queue(maxsize=100_000)
    replay = ExperienceReplay()
    collecting_process = mp.Process(
        target=collect_experience, args=(games, main_net, target_net, queue, params)
    )
    training_process = mp.Process(target=train_net, args=(main_net, queue, replay))

    processes.append(collecting_process)
    processes.append(training_process)
    collecting_process.start()
    training_process.start()

    for p in processes:
        p.join()