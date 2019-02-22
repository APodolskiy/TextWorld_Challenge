import argparse
import logging

from pathlib import Path

import gym
import textworld
from tqdm import tqdm

from agents.baseline_q_learning.custom_agent import BaseQlearningAgent
from agents.utils.params import Params
from test_submission import _validate_requested_infos


def train(game_files):
    logging.basicConfig(level=logging.INFO)
    params = Params.from_file("configs/config.jsonnet")
    agent_params = params.pop("agent")
    train_params = params.pop("training")
    agent = BaseQlearningAgent(agent_params)
    requested_infos = agent.select_additional_infos()
    _validate_requested_infos(requested_infos)

    env_id = textworld.gym.register_games(game_files, requested_infos,
                                          max_episode_steps=agent.max_steps_per_episode,
                                          name="training")
    # env_id = textworld.gym.make_batch(env_id, batch_size=agent.batch_size, parallel=True)
    env = gym.make(env_id)

    for epoch_no in range(1, train_params.pop("n_epochs") + 1):
        stats = {
            "scores": [],
            "steps": [],
        }
        for _ in tqdm(range(len(game_files))):
            obs, infos = env.reset()
            agent.train()

            dones, steps, scores = [False], [], []

            while not all(dones):
                # Increase step counts.
                steps = [step + int(not done) for step, done in zip(steps, dones)]
                commands = agent.act(obs, scores, dones, infos)
                obs, scores, dones, infos = env.step(commands)

            # Let the agent knows the game is done.
            agent.act(obs, scores, dones, infos)

            stats["scores"].extend(scores)
            stats["steps"].extend(steps)

        score = sum(stats["scores"]) / agent.batch_size
        steps = sum(stats["steps"]) / agent.batch_size
        print("Epoch: {:3d} | {:2.1f} pts | {:4.1f} steps".format(epoch_no, score, steps))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train baseline Q-learning agent.")
    parser.add_argument("games", metavar="game", type=str,
                        help="path to the folder with games")
    args = parser.parse_args()

    train_dir = Path(args.games)

    games = [str(f) for f in train_dir.iterdir() if f.is_file() and f.suffix == '.ulx']

    train(games)