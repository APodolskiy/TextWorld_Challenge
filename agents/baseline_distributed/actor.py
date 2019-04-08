import json
from pathlib import Path
from typing import List, Dict, Any

from _jsonnet import evaluate_file
import gym
import spacy
from textworld import EnvInfos
import textworld.gym
from tqdm import tqdm

import torch
import torch.multiprocessing as mp

from agents.baseline_distributed.model import LSTM_DQN
from agents.baseline_distributed.data_utils.preprocess import preprocess, SEP_TOKEN, ITM_TOKEN
from agents.utils.eps_scheduler import LinearScheduler
from agents.utils.generic import _words_to_ids


class Actor(mp.Process):
    def __init__(self,
                 actor_id: int,
                 eps,
                 request_infos,
                 game_files,
                 config,
                 shared_state,
                 shared_replay_memory,
                 shared_writer):
        super(Actor, self).__init__()
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

        self.model = LSTM_DQN(config=self.config["model"], word_vocab=self.word_vocab)
        self.nlp = spacy.load('en', disable=['ner', 'parser', 'tagger'])

        self.eps = eps
        self.eps_scheduler = LinearScheduler(self.config["exploration"])

        self.act_steps = 0
        self.episode_steps = 0
        self.num_episodes = 0
        self.NUM_EPOCHS = 10000

        self._episode_started = False
        self.previous_actions: List[str] = []
        self.scores: List[List[int]] = []
        self.dones: List[List[int]] = []
        self.prev_description_id: List = None
        self.prev_command: List = None

    def run(self):
        env_id = textworld.gym.register_games(self.game_files,
                                              self.request_infos,
                                              max_episode_steps=200,
                                              name='training')
        env_id = textworld.gym.make_batch(env_id, batch_size=1, parallel=False)
        env = gym.make(env_id)
        for epoch_no in range(self.NUM_EPOCHS):
            stats = {
                "scores": [],
                "steps": []
            }
            steps = 0
            for _ in tqdm(range(len(self.game_files))):
                obs, infos = env.reset()
                done, score = False, 0
                while not done:
                    action = self.act(obs[0], infos, score, done)
                    obs, score, done, infos = env.step(action)
                    steps += 1
                stats["scores"].append(score)
                stats["steps"].append(steps)

    def act(self, obs: str, infos: Dict[str, List[Any]], score: int, done: bool) -> str:
        state_description = self.get_game_state_info(obs, infos)
        pass

    def get_game_state_info(self, obs: str, infos: Dict[str, Any]) -> torch.Tensor:
        description_tokens = preprocess(infos["description"][0], "description", tokenizer=self.nlp)
        if len(description_tokens) == 0:
            description_tokens = ["end"]
        description_ids = _words_to_ids(description_tokens, self.word2id)

        inventory_tokens = preprocess(infos["inventory"][0], "inventory", tokenizer=self.nlp)
        inventory_ids = _words_to_ids(inventory_tokens, self.word2id)

        recipe_tokens = preprocess(infos["extra.recipe"][0], "recipe", tokenizer=self.nlp)
        recipe_ids = _words_to_ids(recipe_tokens, self.word2id)

        feedback_tokens = preprocess(obs, "feedback", tokenizer=self.nlp)
        feedback_ids = _words_to_ids(feedback_tokens, self.word2id)

        state_ids = description_ids + [self.SEP_id] + inventory_ids + [self.SEP_id] + recipe_ids + [self.EOS_id]
        input_description = torch.tensor(state_ids, dtype=torch.long)
        return input_description

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
