import textworld
import gym
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from test_submission import _validate_requested_infos
from agents.PGagent.custom_agent import CustomAgent
from agents.PGagent.utils import generate_session


agent = CustomAgent()
# the path for my laptop is /home/nik/Documents/git/textworld/microsoft_starting_kit/sample_games
# and for work pc is /home/nik-96/Documents/git/textworld/microsoft_starting_kit/sample_games
game_dir = Path('/home/nik-96/Documents/git/textworld/microsoft_starting_kit/sample_games')
games = list(game_dir.iterdir())
requested_infos = agent.select_additional_infos()
_validate_requested_infos(requested_infos)

env_id = textworld.gym.register_games(
    # [[str(game) for game in games if str(game).endswith('.ulx')][0]],
    [str(game_dir / 'tw-cooking-recipe1+take1-11Oeig8bSVdGSp78.ulx')],
    requested_infos,
    max_episode_steps=agent.max_nb_steps_per_episode,
    name="training",
)
batch_size = agent.batch_size
if batch_size:
    env_id = textworld.gym.make_batch(env_id, batch_size=batch_size, parallel=True)
else:
    pass
# TODO: add parallel environments
env = gym.make(env_id)
state, info = env.reset()

entropy = []
loss = []
episode_rewards = []

# training loop
for episode in range(500):
    action_probs, rewards = generate_session(agent, env)
    loss_value, entropy_value = agent.update(action_probs, rewards)
    loss.append(loss_value)
    entropy.append(entropy_value)
    episode_rewards.append(np.sum(rewards))
    if episode % 50 == 0:
        print("episode: {}, loss: {}, rewards: {}".format(episode, np.mean(loss[-50:]), episode_rewards[-1]))

agent.save_model('./')

plt.subplot(311)
plt.plot(loss)
plt.title("Loss")
plt.subplot(312)
plt.plot(entropy)
plt.title("Entropy")
plt.subplot(313)
plt.plot(episode_rewards)
plt.title("Episode reward")
plt.show()
