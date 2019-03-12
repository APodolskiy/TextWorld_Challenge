import argparse
import logging

from pathlib import Path

import gym
import textworld
from tqdm import tqdm
import torch.multiprocessing as mp
from agents.baseline_q_learning.advanced_agent import QNet
from agents.baseline_q_learning.custom_agent import BaseQlearningAgent
from agents.utils.params import Params
from test_submission import _validate_requested_infos


def train(game_files, agent, params):
    logging.basicConfig(level=logging.INFO)
    train_params = params.pop("training")
    requested_infos = agent.select_additional_infos()
    _validate_requested_infos(requested_infos)

    env_id = textworld.gym.register_games(
        game_files,
        requested_infos,
        max_episode_steps=agent.max_steps_per_episode,
        name="training",
    )
    batch_size = 2
    env_id = textworld.gym.make_batch(env_id, batch_size=batch_size, parallel=False)
    env = gym.make(env_id)

    for epoch_no in range(1, train_params.pop("n_epochs") + 1):
        stats = {"scores": [], "steps": []}
        for _ in tqdm(range(len(game_files))):
            obs, infos = env.reset()
            agent.train()

            dones = [False for _ in range(batch_size)]
            steps = [0 for _ in range(batch_size)]
            scores = [0 for _ in range(batch_size)]

            while not all(dones):
                # Increase step counts.
                steps = [step + int(not done) for step, done in zip(steps, dones)]
                command = agent.act(obs, scores, dones, infos)
                obs, scores, dones, infos = env.step(command)

        score = sum(stats["scores"])
        steps = sum(stats["steps"])
        print(f"Epoch: {epoch_no:3d} | {score:2.1f} pts | {steps:4.1f} steps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline Q-learning agent.")
    parser.add_argument(
        "games", metavar="game", type=str, help="path to the folder with games"
    )
    args = parser.parse_args()

    train_dir = Path(args.games)

    games = [str(f) for f in train_dir.iterdir() if f.is_file() and f.suffix == ".ulx"]

    params = Params.from_file("configs/config.jsonnet")
    agent_params = params.pop("agent")

    target_net = QNet(agent_params)
    target_net.share_memory()

    value_net = QNet(agent_params)
    value_net.share_memory()

    processes = []

    collecting_process = mp.Process(train, args=())

    for p in processes:
        p.join()
    # train(games)
