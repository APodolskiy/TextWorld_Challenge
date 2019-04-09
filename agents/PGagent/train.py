import textworld
import gym
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
    [[str(game) for game in games if str(game).endswith('.ulx')][0]],
    requested_infos,
    max_episode_steps=50,
    name="training",
)
# batch_size = 16
# env_id = textworld.gym.make_batch(env_id, batch_size=batch_size, parallel=True)
# TODO: add parallel environments
env = gym.make(env_id)
agent = CustomAgent()

# training loop
for episode in range(5):
    states, actions, rewards = generate_session(agent, env)




