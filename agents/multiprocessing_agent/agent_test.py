from argparse import ArgumentParser

import spacy

import torch

import gym
import textworld.gym
from allennlp.common import Params
from textworld import EnvInfos

from agents.multiprocessing_agent.custom_agent import QNet, State, BaseQlearningAgent
from agents.multiprocessing_agent.simple_net import SimpleNet
from agents.multiprocessing_agent.utils import clean_text
from agents.utils.eps_scheduler import EpsScheduler, DeterministicEpsScheduler

required_infos = EnvInfos(
    description=True,
    inventory=True,
    extras=["recipe", "walkthrough"],
    admissible_commands=True,
)


def check_agent(game_file, train_params, agent_net: QNet):
    env_id = textworld.gym.register_games(
        [game_file], required_infos, max_episode_steps=1000, name="check_agent"
    )
    env_id = textworld.gym.make_batch(env_id, batch_size=1, parallel=False)
    env = gym.make(env_id)
    obs, infos = env.reset()
    rewards = [0, ]
    dones = [False]

    actor = BaseQlearningAgent(
        net=agent_net,
        params=train_params,
        eps_scheduler=DeterministicEpsScheduler(),
    )

    print(infos["extra.walkthrough"])

    cnt = 0
    while not all(dones) and cnt < 10:
        infos["gamefile"] = game_file[0]
        infos["is_lost"] = [False]

        commands = actor.act(obs, rewards, dones, infos)
        print(f">{commands[0]}")
        obs, cumulative_rewards, dones, infos = env.step(commands)
        print(obs[0])
        cnt += 1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("game_file", type=str)
    args = parser.parse_args()
    params = Params.from_file("configs/debug_config.jsonnet")
    agent = SimpleNet(device="cpu", tokenizer=spacy.load("en_core_web_sm").tokenizer)
    agent.load_state_dict(torch.load(params["training"]["model_path"]))
    game_file = f"games/train_sample/{args.game_file}"
    check_agent(game_file=game_file, agent_net=agent, train_params=params.pop("training"))