import textworld
import gym
from matplotlib import pyplot as plt
from pathlib import Path
from textworld import EnvInfos
from test_submission import _validate_requested_infos
from agents.PGagent.custom_agent import CustomAgent
from agents.PGagent.utils import generate_session


game_dir = Path('/home/nik-96/Documents/git/textworld/microsoft_starting_kit/sample_games')
games = list(game_dir.iterdir())
requested_infos = EnvInfos(
            max_score=True,
            description=True,
            inventory=True,
            objective=True,
            entities=True,
            command_templates=True,
            extras=["recipe"],
            admissible_commands=True,
        )
_validate_requested_infos(requested_infos)

env_id = textworld.gym.register_games(
    # [[str(game) for game in games if str(game).endswith('.ulx')][0]],
    [str(game_dir / 'tw-cooking-recipe1+take1-11Oeig8bSVdGSp78.ulx')],
    requested_infos,
    max_episode_steps=50,
    name="training",
)
# batch_size = 16
# env_id = textworld.gym.make_batch(env_id, batch_size=batch_size, parallel=True)
# TODO: add parallel environments
env = gym.make(env_id)
agent = CustomAgent()
state, info = env.reset()
entropy = []
loss = []

# training loop
for episode in range(500):
    action_probs, rewards = generate_session(agent, env)
    loss_value, entropy_value = agent.update(action_probs, rewards)
    loss.append(loss_value)
    entropy.append(entropy_value)
    if episode % 50 == 0:
        print("episode: {}, loss: {}".format(episode, loss_value))

plt.subplot(211)
plt.plot(loss)
plt.title("Loss")
plt.subplot(212)
plt.plot(entropy)
plt.title("Entropy")
plt.show()
