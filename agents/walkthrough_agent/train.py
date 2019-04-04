import textworld
import gym
from pathlib import Path
from textworld import EnvInfos
from test_submission import _validate_requested_infos
from textworld.agents import WalkthroughAgent


game_dir = Path('/home/nik-96/Documents/git/textworld/microsoft_starting_kit/sample_games')
games = list(game_dir.iterdir())
requested_infos = EnvInfos(
            max_score=True,
            description=True,
            inventory=True,
            objective=True,
            entities=True,
            command_templates=True,
            extras=["walkthrough", "recipe"],
            admissible_commands=True,
        )
_validate_requested_infos(requested_infos)

env_id = textworld.gym.register_games(
    [str(game) for game in games if str(game).endswith('.ulx')],
    requested_infos,
    max_episode_steps=50,
    name="training",
)
# batch_size = 16
# env_id = textworld.gym.make_batch(env_id, batch_size=batch_size, parallel=False)
env = gym.make(env_id)
for game_no in range(len(games)):
    obs, infos = env.reset()

    done = False
    walk_commands = infos['extra.walkthrough']
    print('' in walk_commands)
    walk_commands = [walk_commands] + [walk_commands[:index] + walk_commands[index+1:]
                                       for index, _ in enumerate(walk_commands)]

    succeeded_commands_seq = []
    for command_set in walk_commands:

        obs, infos = env.reset()
        agent = WalkthroughAgent(commands=command_set)
        agent.reset(env)
        count = 0
        # for iteration in range(100):
        while not done:
            command = agent.act(obs, 0, done)
            print(command)
            if command == '':
                break

            obs, reward, done, infos = env.step(command)
            count += 1
            if done:

                succeeded_commands_seq.append(walk_commands)
                break

    print(len(succeeded_commands_seq))
























