import logging
from multiprocessing.queues import Queue

import gym
import textworld
from tqdm import tqdm

from agents.baseline_q_learning.custom_agent import BaseQlearningAgent
from test_submission import _validate_requested_infos


def collect_experience(game_files, main_net, target_net, buffer: Queue, params):
    logging.basicConfig(level=logging.INFO)
    train_params = params.pop("training")

    actor = BaseQlearningAgent(
        net=target_net, experience_replay_buffer=buffer, config=params.pop("agent")
    )

    requested_infos = actor.select_additional_infos()
    _validate_requested_infos(requested_infos)

    env_id = textworld.gym.register_games(
        game_files,
        requested_infos,
        max_episode_steps=actor.max_steps_per_episode,
        name="training",
    )
    batch_size = 2
    env_id = textworld.gym.make_batch(env_id, batch_size=batch_size, parallel=True)
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

        score = sum(stats["scores"])
        steps = sum(stats["steps"])
        print(f"Epoch: {epoch_no:3d} | {score:2.1f} pts | {steps:4.1f} steps")
    print("Returning")
    return