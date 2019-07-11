from collections import deque
import json
from pathlib import Path
from random import random
from typing import List, Dict, Any, Tuple, Optional

from _jsonnet import evaluate_file
import gym
import numpy as np
import spacy
from textworld import EnvInfos
import textworld.gym
from tqdm import tqdm

import torch
import torch.multiprocessing as mp

from agents.baseline_distributed.model import LSTM_DQN
from agents.baseline_distributed.utils.preprocess import preprocess, SEP_TOKEN, ITM_TOKEN
from agents.baseline_distributed.utils.utils import SimpleTransition
from agents.utils.eps_scheduler import LinearScheduler
from agents.utils.generic import _words_to_ids, pad_sequences


class Actor(mp.Process):
    def __init__(self,
                 actor_id: int,
                 eps: float,
                 request_infos: EnvInfos,
                 game_files: List,
                 config: Dict,
                 shared_state,
                 shared_replay_memory: mp.Queue,
                 shared_writer):
        super(Actor, self).__init__()
        print(f"Creating agent with id: {actor_id}")
        self.word_vocab = []
        self.word2id = {}
        self._load_vocab(vocab_file="./vocab.txt")
        self.EOS_id = self.word2id["</s>"]
        self.SEP_id = self.word2id[SEP_TOKEN]
        self.ITM_id = self.word2id[ITM_TOKEN]

        self.writer = shared_writer
        self.shared_state = shared_state
        self.shared_replay_memory = shared_replay_memory

        self.game_files = game_files
        self.id = actor_id
        self.request_infos = request_infos

        self.config = config
        self.nb_epochs = self.config['training']['nb_epochs']
        self.batch_size = self.config['training']['batch_size']
        self.max_nb_steps_per_episode = self.config['training']['max_nb_steps_per_episode']

        self.model = LSTM_DQN(config=self.config["model"],
                              word_vocab_size=len(self.word_vocab) + 2)
        self.model.load_state_dict(self.shared_state['model'])
        self.nlp = spacy.load('en', disable=['ner', 'parser', 'tagger'])

        self.eps = eps
        self.eps_scheduler = LinearScheduler(self.config["exploration"])

        self.replay_memory = deque(maxlen=50000)

        self.act_steps = 0
        self.episode_steps = 0
        self.num_episodes = 0
        self.NUM_EPOCHS = 10000

        self.SEND_PERIOD = 10

        self._episode_started = False
        self.previous_actions: List[str] = []
        self.scores: List[List[int]] = []
        self.dones: List[List[int]] = []
        self.prev_description_id: Optional[List] = None
        self.prev_command: Optional[List] = None

    def run(self):
        env_id = textworld.gym.register_games(self.game_files,
                                              self.request_infos,
                                              max_episode_steps=100,
                                              name=f'training_{self.id}')
        env = gym.make(env_id)
        steps = 0
        print(f"Start playing with agent: {self.id}")
        for epoch_no in range(self.NUM_EPOCHS):
            stats = {
                "scores": [],
                "steps": []
            }
            prev_command_idx = 0
            for _ in range(len(self.game_files)):
                obs, infos = env.reset()
                done, score = False, 0
                prev_ids = None
                while not done:
                    descriptions, ids, admissible_commands = self.get_state(obs, infos)
                    command_idx = self.act(*descriptions)
                    action = admissible_commands[command_idx]
                    obs, score, done, infos = env.step(action)

                    # Update local replay buffer
                    if prev_ids is not None:
                        transition = SimpleTransition(
                            description_ids=prev_ids[0],
                            command_ids=[prev_ids[1][prev_command_idx]],
                            reward=score,
                            done=done,
                            next_description_ids=ids[0],
                            next_command_ids=ids[1]
                        )
                        self.replay_memory.appendleft(transition)

                    if steps % self.SEND_PERIOD == 0 and steps != 0:
                        for elem in self.replay_memory:
                            self.shared_replay_memory.put(elem)
                        self.replay_memory.clear()

                    prev_ids = ids
                    prev_command_idx = command_idx
                    steps += 1
                stats["scores"].append(score)
                stats["steps"].append(steps)
        print(f"End of running agent {self.id}")

    def get_state(self, obs: str, infos: Dict[str, List[Any]]) -> Tuple[Tuple, Tuple, List]:
        admissible_commands = infos["admissible_commands"]
        state_description, state_ids = self.get_game_state_info(obs, infos)
        commands_description, commands_ids = self.get_commands_description(admissible_commands)
        return (state_description, commands_description), (state_ids, commands_ids), admissible_commands

    def act(self, state_description: torch.Tensor, commands_description: torch.Tensor) -> int:
        # Choose action
        if random() < self.eps:
            chosen_command_idx = np.random.choice(commands_description.size(0))
        else:
            with torch.no_grad():
                q_values = self.model(state_description.unsqueeze(0), [commands_description])[0]
                chosen_command_idx = q_values.argmax().item()
        return chosen_command_idx

    def get_game_state_info(self, obs: str, infos: Dict[str, Any]) -> Tuple[torch.Tensor, List]:
        description_tokens = preprocess(infos["description"], "description", tokenizer=self.nlp)
        if len(description_tokens) == 0:
            description_tokens = ["end"]
        description_ids = _words_to_ids(description_tokens, self.word2id)

        inventory_tokens = preprocess(infos["inventory"], "inventory", tokenizer=self.nlp)
        inventory_ids = _words_to_ids(inventory_tokens, self.word2id)

        recipe_tokens = preprocess(infos["extra.recipe"], "recipe", tokenizer=self.nlp)
        recipe_ids = _words_to_ids(recipe_tokens, self.word2id)

        feedback_tokens = preprocess(obs, "feedback", tokenizer=self.nlp)
        feedback_ids = _words_to_ids(feedback_tokens, self.word2id)

        state_ids = description_ids + [self.SEP_id] + inventory_ids + [self.SEP_id] + recipe_ids + [self.EOS_id]
        input_description = torch.tensor(state_ids, dtype=torch.long)
        return input_description, state_ids

    def get_commands_description(self, commands: List[str]):
        commands_tokens = [preprocess(item, "command", tokenizer=self.nlp) for item in commands]
        commands_ids = [_words_to_ids(tokens, self.word2id) for tokens in commands_tokens]
        commands_ids_pad = pad_sequences(commands_ids).astype('int32')
        commands_description = torch.tensor(commands_ids_pad, dtype=torch.long)
        return commands_description, commands_ids

    # TODO: replace load_vocab method with the argument in constructor
    def _load_vocab(self, vocab_file: str) -> None:
        with open(vocab_file, "r") as fp:
            self.word_vocab = fp.read().split("\n")
        for i, w in enumerate(self.word_vocab + [SEP_TOKEN] + [ITM_TOKEN]):
            self.word2id[w] = i


if __name__ == '__main__':
    train_dir = Path("games/dima_sosat_game")
    games = [str(f) for f in train_dir.iterdir() if f.is_file() and f.suffix == '.ulx']

    def select_additional_infos():
        request_infos = EnvInfos()
        request_infos.description = True
        request_infos.inventory = True
        request_infos.entities, request_infos.verbs = True, True
        request_infos.max_score = True
        request_infos.extras = ["recipe"]
        request_infos.admissible_commands = True
        return request_infos

    actor = Actor(actor_id=0, eps=0.3, request_infos=select_additional_infos(), game_files=games,
                  config=json.loads(evaluate_file("configs/dqn_config.jsonnet")), shared_state=None,
                  shared_replay_memory=None, shared_writer=None)
    actor.run()
