import argparse
import logging
from pathlib import Path
import shutil

import torch
from tqdm import tqdm

from tensorboardX import SummaryWriter

import gym
import textworld.gym
from textworld import EnvInfos

from agents.baseline_dqn.custom_agent import CustomAgent

# List of additional information available during evaluation.
AVAILABLE_INFORMATION = EnvInfos(
    description=True, inventory=True,
    max_score=True, objective=True, entities=True, verbs=True,
    command_templates=True, admissible_commands=True,
    has_won=True, has_lost=True,
    extras=["recipe"]
)


logging.basicConfig(level=logging.INFO)


def _validate_requested_infos(infos: EnvInfos):
    msg = "The following information cannot be requested: {}"
    for key in infos.basics:
        if not getattr(AVAILABLE_INFORMATION, key):
            raise ValueError(msg.format(key))

    for key in infos.extras:
        if key not in AVAILABLE_INFORMATION.extras:
            raise ValueError(msg.format(key))


def train(game_files):
    log_path = Path('runs/test_dqn')
    if log_path.exists():
        shutil.rmtree(str(log_path))
    log_path.mkdir(parents=True)
    writer = SummaryWriter(log_dir=str(log_path))

    agent = CustomAgent(writer=writer)
    requested_infos = agent.select_additional_infos()
    _validate_requested_infos(requested_infos)

    env_id = textworld.gym.register_games(game_files, requested_infos,
                                          max_episode_steps=agent.max_nb_steps_per_episode,
                                          name="training")
    env_id = textworld.gym.make_batch(env_id, batch_size=agent.batch_size, parallel=True)
    env = gym.make(env_id)

    for epoch_no in range(1, agent.nb_epochs + 1):
        stats = {
            "scores": [],
            "steps": [],
        }
        for game_no in tqdm(range(len(game_files))):
            obs, infos = env.reset()
            agent.train()

            scores = [0] * len(obs)
            dones = [False] * len(obs)
            steps = [0] * len(obs)
            while not all(dones):
                # Increase step counts.
                steps = [step + int(not done) for step, done in zip(steps, dones)]
                commands = agent.act(obs, scores, dones, infos)
                obs, scores, dones, infos = env.step(commands)

            # Let the agent knows the game is done.
            agent.act(obs, scores, dones, infos)

            stats["scores"].extend(scores)
            stats["steps"].extend(steps)

        print(f"Scores: {stats['scores']}\nSteps: {stats['steps']}")
        score = sum(stats["scores"]) / agent.batch_size
        steps = sum(stats["steps"]) / agent.batch_size
        print("Epoch: {:3d} | {:2.1f} pts | {:4.1f} steps".format(epoch_no, score, steps))
        writer.add_scalar("score", score, epoch_no)
        if epoch_no % 10 == 0:
            torch.save(agent.model.state_dict(), log_path / "model.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an agent.")
    parser.add_argument("games", metavar="game", type=str,
                        help="path to the folder with games")
    args = parser.parse_args()

    train_dir = Path(args.games)

    games = [str(f) for f in train_dir.iterdir() if f.is_file() and f.suffix == '.ulx']

    train(games)
