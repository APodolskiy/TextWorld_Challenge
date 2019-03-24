"""
Tests trained agent on a provided game
"""
from argparse import ArgumentParser

import gym
import spacy
import textworld.gym
import torch
from allennlp.common import Params

from agents.DRQN.custom_agent import BaseQlearningAgent
from agents.DRQN.networks.simple_net import SimpleNet
from agents.utils.eps_scheduler import DeterministicEpsScheduler


def check_agent(game_file, train_params, agent_net):
    env_id = textworld.gym.register_games(
        [game_file],
        BaseQlearningAgent.select_additional_infos(),
        max_episode_steps=1000,
        name="check_agent",
    )
    env_id = textworld.gym.make_batch(env_id, batch_size=1, parallel=False)
    env = gym.make(env_id)
    obs, infos = env.reset()
    cumulative_rewards = [0]
    dones = [False]
    actor = BaseQlearningAgent(
        net=agent_net, params=train_params, eps_scheduler=DeterministicEpsScheduler()
    )
    actor.start_episode(infos)

    print(infos["extra.walkthrough"])

    cnt = 0
    while not all(dones) and cnt < 20:
        infos["gamefile"] = game_file[0]
        commands = actor.act(obs, cumulative_rewards, dones, infos)
        print(f">{commands[0]}")
        obs, cumulative_rewards, dones, infos = env.step(commands)
        print(obs[0])
        cnt += 1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("game_file", type=str)
    args = parser.parse_args()
    params = Params.from_file("configs/debug_config.jsonnet")
    agent = SimpleNet(
        config=params["network"],
        device="cpu",
        vocab_size=params["training"]["vocab_size"],
    )
    agent.load_state_dict(torch.load(params["training"]["model_path"]))
    game_file = f"games/train_sample/{args.game_file}"
    check_agent(
        game_file=game_file, agent_net=agent, train_params=params.pop("training")
    )
