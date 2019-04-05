"""
Evaluates agent
"""
from argparse import ArgumentParser
from pathlib import Path

import gym
import numpy
import spacy
import textworld.gym
import torch
from allennlp.common import Params

from agents.DRQN.custom_agent import BaseQlearningAgent
from agents.DRQN.networks.simple_net import SimpleNet
from agents.DRQN.policy.policies import GreedyPolicy
from agents.utils.eps_scheduler import DeterministicEpsScheduler
from agents.utils.logging import get_sample_history_trace


def check_agent(game_files, train_params, agent_net, batch_size=1):
    env_id = textworld.gym.register_games(
        game_files,
        BaseQlearningAgent.select_additional_infos(),
        max_episode_steps=100,
        name="check_agent",
    )
    env_id = textworld.gym.make_batch(env_id, batch_size=batch_size, parallel=False)
    env = gym.make(env_id)
    actor = BaseQlearningAgent(
        net=agent_net, params=train_params, policy=GreedyPolicy()
    )
    for i in range(len(game_files)):
        game_file = game_files[i]
        obs, infos = env.reset()
        cumulative_rewards = [0] * batch_size
        dones = [False] * batch_size
        actor.start_episode(infos)
        # print(infos["extra.walkthrough"])
        cnt = 0
        while not all(dones):
            infos["gamefile"] = game_file
            commands = actor.act(obs, cumulative_rewards, dones, infos)
            obs, cumulative_rewards, dones, infos = env.step(commands)
            cnt += 1
        infos["gamefile"] = game_file
        actor.act(obs, cumulative_rewards, dones, infos)
        mean_reward = numpy.mean(cumulative_rewards)
        max_reward = infos["max_score"][0]
        print(f"Scored {mean_reward}/{max_reward}")
        # print(get_sample_history_trace(actor.history, game_file))


def get_ulx_filenames(dir_path):
    return [
        str(f) for f in Path(dir_path).iterdir() if f.is_file() and f.suffix == ".ulx"
    ]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("games_dir", type=str)
    args = parser.parse_args()
    params = Params.from_file("configs/config.jsonnet")
    agent = SimpleNet(
        config=params["network"],
        device="cpu",
        vocab_size=params["training"]["vocab_size"],
    )
    agent.load_state_dict(torch.load(params["training"]["model_path"]))

    check_agent(
        game_files=get_ulx_filenames(args.games_dir),
        agent_net=agent,
        train_params=params.pop("training"),
        batch_size=10
    )
