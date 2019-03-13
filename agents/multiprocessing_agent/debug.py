import logging
from multiprocessing import Queue
from pathlib import Path

import gym
import textworld
from tqdm import tqdm

from agents.multiprocessing_agent.learning import learn
from agents.utils.params import Params
from agents.multiprocessing_agent.custom_agent import BaseQlearningAgent, QNet
from agents.utils.replay import ExperienceReplay
from test_submission import _validate_requested_infos

logging.basicConfig(level=logging.INFO)


def debug(game_files, buffer, params, policy_net, target_net):
    train_params = params.pop("training")
    actor = BaseQlearningAgent(
        net=target_net, experience_replay_buffer=buffer, config=params.pop("agent")
    )
    replay = ExperienceReplay()
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
                obs, scores, dones, infos = env.step(command)

        learn(net=policy_net, target_net=target_net, replay_buffer=replay, queue=buffer)


if __name__ == "__main__":
    games = [
        str(f)
        for f in Path("games/train_sample").iterdir()
        if f.is_file() and f.suffix == ".ulx"
    ]
    params = Params.from_file("configs/config.jsonnet")

    my_net= QNet(params.get("agent").get("network")).cuda()

    debug(
        buffer=Queue(),
        params=params,
        game_files=games[:4],
        target_net=my_net,
        policy_net=my_net
    )
