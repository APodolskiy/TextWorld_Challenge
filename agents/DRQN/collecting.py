from multiprocessing import Queue

import torch
from logging import info

import gym
import textworld.gym
import numpy
from tqdm import tqdm
from agents.DRQN.custom_agent import BaseQlearningAgent
from tensorboardX import SummaryWriter

collecting_step = 1


def collect_experience(
    game_files,
    buffer: Queue,
    train_params,
    eps_scheduler,
    target_net,
    policy_net,
    log_dir,
    env=None,
):
    global collecting_step
    if log_dir is not None:
        writer = SummaryWriter(log_dir)
    actor = BaseQlearningAgent(
        net=policy_net, params=train_params, eps_scheduler=eps_scheduler
    )
    batch_size = train_params.pop("n_parallel_envs")
    if env is None:
        env_id = textworld.gym.register_games(
            game_files,
            BaseQlearningAgent.select_additional_infos(),
            max_episode_steps=actor.max_steps_per_episode,
            name="training_par",
        )
        env_id = textworld.gym.make_batch(
            env_id,
            batch_size=batch_size,
            parallel=train_params.pop("use_separate_process_envs"),
        )
        env = gym.make(env_id)

    update_freq = train_params.pop("target_net_update_freq")
    for epoch_no in range(1, train_params.pop("n_epochs_collection") + 1):
        for _ in tqdm(range(1, len(game_files) + 1)):
            if collecting_step % update_freq == 0:
                info("Updating target net")
                target_net.load_state_dict(policy_net.state_dict())

            obs, infos = env.reset()
            actor.train()

            dones = [False for _ in range(batch_size)]
            steps = [0 for _ in range(batch_size)]
            cumulative_rewards = [0 for _ in range(batch_size)]

            while not all(dones):
                steps = [step + int(not done) for step, done in zip(steps, dones)]

                # TODO: only one game is supported
                infos["gamefile"] = game_files[0]
                with torch.no_grad():
                    command = actor.act(obs, cumulative_rewards, dones, infos)
                obs, cumulative_rewards, dones, infos = env.step(command)
            infos["gamefile"] = game_files[0]
            actor.act(obs, cumulative_rewards, dones, infos)
            assert all([actor.history[i][-1].done for i in range(actor.batch_size)])
            for game in actor.history.values():
                buffer.put(game)
            actor.reset()
            actor.eps_scheduler.increase_step()
            if log_dir is not None:
                writer.add_scalar(
                    "train/avg_reward", numpy.mean(cumulative_rewards), collecting_step
                )
                writer.add_scalar("train/avg_steps", numpy.mean(steps), collecting_step)
                writer.add_scalar("train/eps", actor.eps_scheduler.eps, collecting_step)
                writer.add_histogram(
                    "train/actor_target_net_weights",
                    target_net.hidden_to_scores.weight.clone().detach().cpu().numpy(),
                    collecting_step,
                )
                writer.add_histogram(
                    "train/actor_policy_net_weights",
                    policy_net.hidden_to_scores.weight.clone().detach().cpu().numpy(),
                    collecting_step,
                )
            collecting_step += 1