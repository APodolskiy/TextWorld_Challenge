import gym
import textworld
from tqdm import tqdm
from agents.multiprocessing_agent.custom_agent import BaseQlearningAgent
from test_submission import _validate_requested_infos


def collect_experience(game_files, buffer, params, target_net):
    train_params = params.pop("training")
    actor = BaseQlearningAgent(
        net=target_net, experience_replay_buffer=buffer, config=train_params
    )
    requested_infos = actor.select_additional_infos()
    _validate_requested_infos(requested_infos)

    env_id = textworld.gym.register_games(
        game_files,
        requested_infos,
        max_episode_steps=actor.max_steps_per_episode,
        name="training",
    )
    batch_size = train_params.pop("n_parallel_envs")
    env_id = textworld.gym.make_batch(env_id, batch_size=batch_size, parallel=False)
    env = gym.make(env_id)

    for epoch_no in range(1, train_params.pop("n_epochs") + 1):
        for _ in tqdm(range(len(game_files))):
            obs, infos = env.reset()
            actor.train()

            dones = [False for _ in range(batch_size)]
            steps = [0 for _ in range(batch_size)]
            cumulative_rewards = None
            prev_cumulative_rewards = [0 for _ in range(batch_size)]
            rewards = prev_cumulative_rewards

            while not all(dones):
                steps = [step + int(not done) for step, done in zip(steps, dones)]
                command = actor.act(obs, rewards, dones, infos)
                obs, cumulative_rewards, dones, infos = env.step(command)

                rewards = [
                    c_r - p_c_r
                    for c_r, p_c_r in zip(cumulative_rewards, prev_cumulative_rewards)
                ]
                prev_cumulative_rewards = cumulative_rewards
            print(cumulative_rewards)
