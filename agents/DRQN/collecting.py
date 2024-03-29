from logging import info
from multiprocessing import Queue
from pathlib import Path

import gym
import numpy
import textworld.gym
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from agents.DRQN.custom_agent import BaseQlearningAgent
from agents.utils.logging import get_sample_history_trace

collecting_step = 1
current_plot_names = set()

def collect_experience(
    game_files,
    buffer: Queue,
    train_params,
    policy,
    target_net,
    policy_net,
    log_dir,
    env=None,
):
    global collecting_step
    max_plots = 30
    global current_plot_names
    if log_dir is not None:
        writer = SummaryWriter(log_dir)
    actor = BaseQlearningAgent(net=policy_net, params=train_params, policy=policy)
    batch_size = train_params.pop("n_parallel_envs")
    use_parallel_envs = train_params.pop("use_separate_process_envs")
    if env is None:
        env_id = textworld.gym.register_games(
            game_files,
            BaseQlearningAgent.select_additional_infos(),
            max_episode_steps=actor.max_steps_per_episode,
            name="training_par",
        )
        env_id = textworld.gym.make_batch(
            env_id, batch_size=batch_size, parallel=use_parallel_envs
        )
        env = gym.make(env_id)

    update_freq = train_params.pop("target_net_update_freq")
    for epoch_no in range(1, train_params.pop("n_epochs_collection") + 1):
        n_games_to_play = min(train_params.pop("n_played_games"), len(game_files))
        for _ in tqdm(range(1, n_games_to_play + 1)):
            if collecting_step % update_freq == 0:
                info("Updating target net")
                target_net.load_state_dict(policy_net.state_dict())

            obs, infos = env.reset()
            if use_parallel_envs:
                env.envs[0].get("env.current_gamefile")
                current_gamefile = env.envs[0].result()
                env.envs[1].get("env.current_gamefile")
                assert current_gamefile == env.envs[1].result()
            else:
                current_gamefile = env.envs[0].env.current_gamefile
            current_gamefile = Path(current_gamefile).name

            actor.start_episode(infos)
            actor.train()

            dones = [False for _ in range(batch_size)]
            steps = [0 for _ in range(batch_size)]
            cumulative_rewards = [0 for _ in range(batch_size)]

            while not all(dones):
                steps = [step + int(not done) for step, done in zip(steps, dones)]

                # TODO: only one game is supported
                infos["gamefile"] = current_gamefile
                with torch.no_grad():
                    command = actor.act(obs, cumulative_rewards, dones, infos)
                obs, cumulative_rewards, dones, infos = env.step(command)
            infos["gamefile"] = current_gamefile
            actor.act(obs, cumulative_rewards, dones, infos)
            assert all(
                [actor.history[i][-1].transition.done for i in range(actor.batch_size)]
            )
            for game in actor.history.values():
                buffer.put([item.transition for item in game])
            if log_dir is not None:
                if current_gamefile in current_plot_names or len(current_plot_names) <= max_plots:
                    current_plot_names.add(current_gamefile)
                    writer.add_scalar(
                        f"avg_reward/{current_gamefile}",
                        numpy.mean(cumulative_rewards),
                        collecting_step,
                    )
                    writer.add_scalar(
                        f"avg_steps/{current_gamefile}", numpy.mean(steps), collecting_step
                    )

                    # if collecting_step % 5 == 0:
                    with open(log_dir / f"game_{collecting_step}.txt", "w") as f:
                        trace = get_sample_history_trace(
                            actor.history, current_gamefile
                        )
                        print(trace, file=f)
            collecting_step += 1
            actor.reset()
