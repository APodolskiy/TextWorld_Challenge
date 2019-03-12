import logging
from multiprocessing import Queue
from pathlib import Path

import gym
import textworld
from tqdm import tqdm

from agents.utils.params import Params
from agents.multiprocessing_agent.custom_agent import BaseQlearningAgent, QNet
from test_submission import _validate_requested_infos

logging.basicConfig(level=logging.INFO)


def debug(game_files, buffer, params, target_net):
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
    env_id = textworld.gym.make_batch(env_id, batch_size=batch_size, parallel=False)
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

                actor.net(obs, infos["admissible_commands"], infos)

                obs, scores, dones, infos = env.step(command)


if __name__ == "__main__":
    games = [
        str(f)
        for f in Path("games/train_sample").iterdir()
        if f.is_file() and f.suffix == ".ulx"
    ]
    params = Params.from_file("configs/config.jsonnet")
    debug(
        buffer=Queue(),
        params=params,
        game_files=games,
        target_net=QNet(params.get("agent").get("network")).cuda()
    )
