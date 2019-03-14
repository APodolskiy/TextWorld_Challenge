import torch
from logging import info

import gym
import numpy
import textworld
from tqdm import tqdm
from agents.multiprocessing_agent.custom_agent import BaseQlearningAgent, QNet
from test_submission import _validate_requested_infos
from tensorboardX import SummaryWriter


def collect_experience(
    game_files,
    buffer,
    train_params,
    eps_scheduler_params,
    target_net: QNet,
    policy_net: QNet,
    log_dir,
):
    if log_dir is not None:
        writer = SummaryWriter(log_dir)
    actor = BaseQlearningAgent(
        net=policy_net,
        experience_replay_buffer=buffer,
        params=train_params,
        eps_scheduler_params=eps_scheduler_params,
    )
    requested_infos = actor.select_additional_infos()
    _validate_requested_infos(requested_infos)

    env_id = textworld.gym.register_games(
        game_files,
        requested_infos,
        max_episode_steps=actor.max_steps_per_episode,
        name="training_par",
    )
    batch_size = train_params.pop("n_parallel_envs")
    env_id = textworld.gym.make_batch(env_id, batch_size=batch_size, parallel=False)
    env = gym.make(env_id)

    global_step = 0

    update_freq = train_params.pop("target_net_update_freq")
    for epoch_no in range(1, train_params.pop("n_epochs_collection") + 1):
        for step in tqdm(range(1, len(game_files) + 1)):

            if (global_step + 1) % update_freq == 0:
                info("Updating target net")
                target_net.load_state_dict(policy_net.state_dict())

            obs, infos = env.reset()
            actor.train()

            dones = [False for _ in range(batch_size)]
            steps = [0 for _ in range(batch_size)]
            cumulative_rewards = None
            prev_cumulative_rewards = [0 for _ in range(batch_size)]
            rewards = prev_cumulative_rewards

            while not all(dones):
                steps = [step + int(not done) for step, done in zip(steps, dones)]

                infos["is_lost"] = [
                    ("You lost!" in o if d else False) for o, d in zip(obs, dones)
                ]
                with torch.no_grad():
                    command = actor.act(obs, rewards, dones, infos)
                obs, cumulative_rewards, dones, infos = env.step(command)

                rewards = [
                    c_r - p_c_r
                    for c_r, p_c_r in zip(cumulative_rewards, prev_cumulative_rewards)
                ]
                prev_cumulative_rewards = cumulative_rewards
            actor.reset()
            global_step += 1
            actor.eps_scheduler.increase_step()
            if log_dir is not None:
                writer.add_scalar(
                    "train/avg_reward", numpy.mean(cumulative_rewards), global_step
                )
                writer.add_scalar(
                    "train/avg_steps", numpy.mean(steps), global_step
                )
                writer.add_scalar("train/eps", actor.eps_scheduler.eps, global_step)
                writer.add_histogram(
                    "train/actor_target_net_weights",
                    target_net.hidden_to_scores.weight.clone().detach().cpu().numpy(),
                    global_step,
                )
                writer.add_histogram(
                    "train/actor_policy_net_weights",
                    policy_net.hidden_to_scores.weight.clone().detach().cpu().numpy(),
                    global_step,
                )
