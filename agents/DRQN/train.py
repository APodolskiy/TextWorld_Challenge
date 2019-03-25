import logging
import os
from multiprocessing import Queue
from pathlib import Path
from shutil import rmtree

from agents.utils.params import Params

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    games = [
        str(f)
        for f in Path("games/train_sample").iterdir()
        if f.is_file() and f.suffix == ".ulx"
    ][:1]

    logging.info(games)

    if os.environ.get("DEBUG_MODE") != "1":
        params = Params.from_file("configs/config.jsonnet")
        logging.info("Starting in TRAIN mode")
        log_dir = os.environ.get("EXP_NAME", None)

        if log_dir is not None and log_dir != "":
            log_dir = Path(log_dir)
            force = os.environ.get("FORCE_OVERWRITE_LOGS", False)
            if log_dir.exists():
                if not int(force) == 1:
                    raise RuntimeError("Already exists, aborting")
                rmtree(log_dir)
            log_dir.mkdir(parents=True)
    else:
        params = Params.from_file("configs/debug_config.jsonnet")
        logging.info("Starting in DEBUG mode")
        log_dir = None

    from agents.DRQN.collecting import collect_experience
    from agents.DRQN.custom_agent import BaseQlearningAgent
    from agents.DRQN.learning import learn
    from agents.DRQN.networks.simple_net import SimpleNet
    from agents.utils.eps_scheduler import EpsScheduler
    from agents.utils.replay import BinaryPrioritizeReplayMemory, SeqTernaryPrioritizeReplayMemory
    import textworld.gym
    import gym

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
    vocab_size = params["training"].pop("vocab_size")
    my_net = SimpleNet(
        config=params["network"].duplicate(), device=actor_device, vocab_size=vocab_size
    ).to(actor_device)
    learner_device = params["training"].pop("learner_device")
    target_net = SimpleNet(
        config=params["network"], device=learner_device, vocab_size=vocab_size
    ).to(learner_device)
    target_net.load_state_dict(my_net.state_dict())
    queue = Queue()

    replay_memory_params = params.pop("replay_memory")
    replay_buffer = SeqTernaryPrioritizeReplayMemory(
        capacity=replay_memory_params.pop("capacity"),
        priority_fraction=replay_memory_params.pop("priority_fraction"),
    )

    train_params = params.pop("training")
    eps_params = params.pop("epsilon")

    eps_scheduler = EpsScheduler(eps_params)
    for _ in range(1000):
        collect_experience(
            buffer=queue,
            train_params=train_params.duplicate(),
            eps_scheduler=eps_scheduler,
            game_files=games,
            target_net=target_net,
            policy_net=my_net,
            log_dir=log_dir,
            env=env,
        )

        learn(
            policy_net=my_net,
            target_net=target_net,
            replay_buffer=replay_buffer,
            queue=queue,
            params=train_params.duplicate(),
            log_dir=log_dir,
        )
