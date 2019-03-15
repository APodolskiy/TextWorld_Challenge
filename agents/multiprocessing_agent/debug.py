import logging
from multiprocessing import Queue
from pathlib import Path

import spacy

from agents.multiprocessing_agent.collecting import collect_experience
from agents.multiprocessing_agent.learning import learn
from agents.multiprocessing_agent.simple_net import SimpleNet, SimpleBowNet
from agents.utils.params import Params
from agents.multiprocessing_agent.custom_agent import QNet
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
    my_net = SimpleBowNet(
        device=actor_device, tokenizer=spacy.load("en_core_web_sm").tokenizer
    ).to(actor_device)
    learner_device = params["training"].pop("learner_device")
    target_net = SimpleBowNet(
        tokenizer=spacy.load("en_core_web_sm").tokenizer, device=learner_device
    ).to(learner_device)
    target_net.load_state_dict(my_net.state_dict())
    queue = Queue()

    replay_buffer = BinaryPrioritizeReplayMemory(capacity=500000, priority_fraction=0.5)

    train_params = params.pop("training")
    eps_params = params.pop("epsilon")

    for _ in range(1000):
        collect_experience(
            buffer=queue,
            train_params=train_params.duplicate(),
            eps_scheduler_params=eps_params.duplicate(),
            game_files=games,
            target_net=target_net,
            policy_net=my_net,
            log_dir="debug_runs/actor",
        )

        learn(
            policy_net=my_net,
            target_net=target_net,
            replay_buffer=replay_buffer,
            queue=queue,
            params=train_params.duplicate(),
            log_dir="debug_runs/learner_1e-4",
        )
