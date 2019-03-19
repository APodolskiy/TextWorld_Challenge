from argparse import ArgumentParser

import spacy

import torch

import gym
import textworld.gym
from allennlp.common import Params
from textworld import EnvInfos

from agents.multiprocessing_agent.custom_agent import QNet, State
from agents.multiprocessing_agent.simple_net import SimpleNet

required_infos = EnvInfos(
    description=True, inventory=True, extras=["recipe", "walkthrough"], admissible_commands=True
)


def check_agent(game_file, agent: QNet):
    env_id = textworld.gym.register_games(
        [game_file], required_infos, max_episode_steps=1000, name="check_agent"
    )
    env_id = textworld.gym.make_batch(env_id, batch_size=1, parallel=False)
    env = gym.make(env_id)
    obs, infos = env.reset()
    print(infos["extra.walkthrough"])
    adm_commands = infos["admissible_commands"]

    q_values = agent(
        [
            State(inventory=inv, description=desc, feedback=o)
            for o, desc, inv in zip(obs, infos["description"], infos["inventory"])
        ],
        adm_commands,
    )
    q_max = q_values[0].max()

    for command, q_value in zip(adm_commands[0], q_values[0]):
        print(f"{command:35}{'*' if q_value == q_max else ''} -> {q_value.item()}")
    for _ in range(10):
        obs, cumulative_rewards, dones, infos = env.step(
            [adm_commands[0][q_values[0].argmax().item()]]
        )
        print(obs)
        adm_commands = infos["admissible_commands"]
        q_values = agent(
            [
                State(inventory=inv, description=desc, feedback=o)
                for o, desc, inv in zip(obs, infos["description"], infos["inventory"])
            ],
            adm_commands,
        )
        q_max = q_values[0].max()

        for command, q_value in zip(adm_commands[0], q_values[0]):
            print(f"{command:35}{'*' if q_value == q_max else ''} -> {q_value.item()}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("game_file", type=str)
    args = parser.parse_args()
    params = Params.from_file("configs/debug_config.jsonnet")
    agent = SimpleNet(device="cpu", tokenizer=spacy.load("en_core_web_sm").tokenizer)
    agent.load_state_dict(torch.load(params["training"]["model_path"]))
    game_file = (
        f"games/train_sample/{args.game_file}"
    )
    check_agent(game_file, agent)
