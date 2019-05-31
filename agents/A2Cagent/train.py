import textworld
import gym
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from tqdm import trange
from test_submission import _validate_requested_infos
from agents.A2Cagent.custom_agent import CustomAgent
from tensorboardX import SummaryWriter

agent = CustomAgent()

# the path for my laptop is /home/nik/Documents/git/textworld/microsoft_starting_kit/sample_games
# and for work pc is /home/nik-96/Documents/git/textworld/microsoft_starting_kit/sample_games
game_dir = Path('/home/nik-96/Documents/git/textworld/microsoft_starting_kit/sample_games')
games = list(game_dir.iterdir())
requested_infos = agent.select_additional_infos()
_validate_requested_infos(requested_infos)

env_id = textworld.gym.register_games(
    # [[str(game) for game in games if str(game).endswith('.ulx')][0]],
    #
    [str(game_dir / 'tw-cooking-recipe1+take1-11Oeig8bSVdGSp78.ulx')],
    requested_infos,
    max_episode_steps=agent.max_nb_steps_per_episode,
    name="training",
)
batch_size = agent.batch_size
if batch_size:
    env_id = textworld.gym.make_batch(env_id, batch_size=batch_size, parallel=True)

env = gym.make(env_id)


print("[INFO] training process")
# in directory of that file run tensorboard logdir='./'
writer = SummaryWriter()

EPOCHS = int(1e6)

# logging
now = datetime.now()
time_name = f'0{now.month}{now.day}_{now.hour}:{now.minute if now.minute > 9 else "0" + str(now.minute)}'
with open('./logs/' + time_name, 'w') as f:
    for property, value in agent.params.items():
        f.write(f'{property}: {value}\n')
    f.write(f'epochs: {EPOCHS}')

# training
states, infos = env.reset()

for episode in trange(EPOCHS):
    actions, actions_probs = agent.act(states, infos)
    next_states, rewards, dones, infos = env.step(actions)
    actor_loss, entropy_value, critic_loss = agent.update(actions_probs, states, next_states, rewards)
    states = next_states
    if all(dones):
        states, infos = env.reset()

    # tensorboardX stuff
    if episode % 100 == 0:
        writer.add_scalar("actor loss", actor_loss, episode)
        writer.add_scalar("critic loss", critic_loss, episode)
        writer.add_scalar("reward", np.mean(rewards), episode)
        writer.add_scalar("entropy", entropy_value, episode)
        writer.add_scalar("won games", sum(infos["has_won"]), episode)
        writer.add_scalar("lost games", sum(infos["has_lost"]), episode)
        # TODO: get correct distribution
        writer.add_histogram("rewards batch distributions", np.sum(rewards, axis=0))
    if episode % 1000 == 0:
        if episode != 0:
            # model checkpoints saving
            os.mkdir(os.path.dirname(os.path.abspath(__file__)) + '/models' + f'/{time_name}_episode_{episode}')
            agent.save_model(os.path.dirname(os.path.abspath(__file__)) + '/models' + f'/{time_name}_episode_{episode}')

