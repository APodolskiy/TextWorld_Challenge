import logging
from os.path import exists
from shutil import rmtree

import gym
import spacy
from multiprocessing import Queue
from pathlib import Path

import spacy
import textworld.gym
from agents.multiprocessing_agent.collecting import collect_experience
from agents.multiprocessing_agent.custom_agent import BaseQlearningAgent
from agents.multiprocessing_agent.learning import learn
from agents.multiprocessing_agent.simple_net import SimpleNet, SimpleBowNet
from agents.utils.eps_scheduler import EpsScheduler
from agents.utils.params import Params
from agents.multiprocessing_agent.bert_net import QNet
from agents.utils.replay import BinaryPrioritizeReplayMemory

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    games = [
        str(f)
        for f in Path("games/train_sample").iterdir()
        if f.is_file() and f.suffix == ".ulx"
    ][:1]

    params = Params.from_file("configs/debug_config.jsonnet")
    actor_device = params["training"].pop("actor_device")
    requested_infos = BaseQlearningAgent.select_additional_infos()
    env_id = textworld.gym.register_games(
        games,
        requested_infos,
        max_episode_steps=params["training"]["max_steps_per_episode"],
        name="training_par",
    )
    env_id = textworld.gym.make_batch(
        env_id,
        batch_size=params["training"]["n_parallel_envs"],
        parallel=params["training"]["use_separate_process_envs"],
    )
    env = gym.make(env_id)

    my_net = SimpleNet(
        device=actor_device, tokenizer=spacy.load("en_core_web_sm").tokenizer
    ).to(actor_device)
    learner_device = params["training"].pop("learner_device")
    target_net = SimpleNet(
        tokenizer=spacy.load("en_core_web_sm").tokenizer, device=learner_device
    ).to(learner_device)
    target_net.load_state_dict(my_net.state_dict())
    queue = Queue()

    replay_memory_params = params.pop("replay_memory")
    replay_buffer = BinaryPrioritizeReplayMemory(
        capacity=replay_memory_params.pop("capacity"),
        priority_fraction=replay_memory_params.pop("priority_fraction"),
    )

    train_params = params.pop("training")
    eps_params = params.pop("epsilon")

    log_dir = "debug_runs"
    if exists(log_dir):
        rmtree(log_dir)
        Path(log_dir).mkdir(parents=True)
    eps_scheduler = EpsScheduler(eps_params)
    for _ in range(1000):
        collect_experience(
            buffer=queue,
            train_params=train_params.duplicate(),
            eps_scheduler=eps_scheduler,
            game_files=games,
            target_net=target_net,
            policy_net=my_net,
            log_dir="debug_runs/actor",
            env=env
        )

        learn(
            policy_net=my_net,
            target_net=target_net,
            replay_buffer=replay_buffer,
            queue=queue,
            params=train_params.duplicate(),
            log_dir="debug_runs/learner",
        )
