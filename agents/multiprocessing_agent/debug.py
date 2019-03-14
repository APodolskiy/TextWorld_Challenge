import logging
from multiprocessing import Queue
from pathlib import Path

from agents.multiprocessing_agent.collecting import collect_experience
from agents.multiprocessing_agent.learning import learn
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
    my_net = QNet(params.get("network"), actor_device).to(actor_device)
    queue = Queue()
    collect_experience(
        buffer=queue,
        train_params=params.get("training"),
        eps_scheduler_params=params.pop("epsilon"),
        game_files=games,
        target_net=my_net,
        policy_net=my_net,
        log_dir=None
    )
    replay_buffer = BinaryPrioritizeReplayMemory(capacity=500000, priority_fraction=0.5)
    learn(
        policy_net=my_net,
        target_net=my_net,
        replay_buffer=replay_buffer,
        queue=queue,
        params=params.pop("training"),
        log_dir=None
    )
