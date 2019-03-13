import logging
from multiprocessing import Queue
from pathlib import Path

from agents.multiprocessing_agent.collecting import collect_experience
from agents.multiprocessing_agent.learning import learn
from agents.utils.params import Params
from agents.multiprocessing_agent.custom_agent import QNet
from agents.utils.replay import ExperienceReplay

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    games = [
        str(f)
        for f in Path("games/train_sample").iterdir()
        if f.is_file() and f.suffix == ".ulx"
    ]
    params = Params.from_file("configs/config.jsonnet")

    my_net = QNet(params.get("network")).cuda()

    queue = Queue()

    collect_experience(
        buffer=queue,
        params=params,
        game_files=games,
        target_net=my_net,
    )
    replay_buffer = ExperienceReplay()
    learn(net=my_net, target_net=my_net, replay_buffer=replay_buffer, queue=queue)
