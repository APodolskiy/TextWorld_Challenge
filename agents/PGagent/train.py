import textworld
import gym
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from tqdm import trange
from test_submission import _validate_requested_infos
from agents.PGagent.custom_agent import CustomAgent
from agents.PGagent.utils import generate_session
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
state, info = env.reset()

print("[INFO] training process")
# in directory of that file run tensorboard logdir='./'
writer = SummaryWriter()

EPOCHS = 1000

# logging
now = datetime.now()
logs_filename = f'./logs/0{now.month}{now.day}_{now.hour}:{now.minute if now.minute > 9 else "0" + str(now.minute)}'
with open(logs_filename, 'w') as f:
    for property, value in agent.params.items():
        f.write(f'{property}: {value}\n')
    f.write(f'epochs: {EPOCHS}')

os.mkdir('./models')

# training loop
for episode in trange(1000):
    actions_probs, rewards, won_games, lost_games = generate_session(agent, env)
    # action_probs and rewards shape [episode_length, batch_size]
    loss_value, entropy_value = agent.update(actions_probs, rewards)
    # tensorboardX stuff
    writer.add_scalar("loss", loss_value, episode)
    writer.add_scalar("reward", np.mean(np.sum(rewards, axis=0)), episode)
    writer.add_scalar("entropy", entropy_value, episode)
    writer.add_scalar("won games", won_games, episode)
    writer.add_scalar("lost games", lost_games, episode)
    # TODO: get correct distribution
    writer.add_histogram("rewards batch distributions", np.sum(rewards, axis=0))
    if episode % 5 == 0:
        if episode != 0:
            # model checkpoints saving
            os.mkdir(os.path.dirname(os.path.abspath(__file__)) + '/models' + f'/{episode}_episode')
            agent.save_model(os.path.dirname(os.path.abspath(__file__)) + '/models' + f'/{episode}_episode')
