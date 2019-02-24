import argparse
import logging
from pathlib import Path
from tqdm import tqdm

import gym
import textworld.gym
from textworld import EnvInfos

from agents.lstm_dqn.custom_agent import CustomAgent

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

    agent = CustomAgent()
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
        import time
        e_time = time.time()
        for game_no in tqdm(range(len(game_files))):
            g_time = time.time()
            obs, infos = env.reset()
            agent.train()

            scores = [0] * len(obs)
            dones = [False] * len(obs)
            steps = [0] * len(obs)
            while not all(dones):
                # Increase step counts.
                steps = [step + int(not done) for step, done in zip(steps, dones)]
                commands = agent.act(obs, scores, dones, infos)
                s_time = time.time()
                obs, scores, dones, infos = env.step(commands)
                print(f"Time to step {time.time() - s_time}")

            # Let the agent knows the game is done.
            agent.act(obs, scores, dones, infos)

            stats["scores"].extend(scores)
            stats["steps"].extend(steps)
            print(f"Time for the game: {time.time() - g_time} steps: {steps}")

        score = sum(stats["scores"]) / agent.batch_size
        steps = sum(stats["steps"]) / agent.batch_size
        print("Epoch: {:3d} | {:2.1f} pts | {:4.1f} steps".format(epoch_no, score, steps))
        print(f"Epoch time: {time.time() - e_time}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an agent.")
    parser.add_argument("games", metavar="game", type=str,
                        help="path to the folder with games")
    args = parser.parse_args()

    train_dir = Path(args.games)

    games = [str(f) for f in train_dir.iterdir() if f.is_file() and f.suffix == '.ulx']

    train(games)
