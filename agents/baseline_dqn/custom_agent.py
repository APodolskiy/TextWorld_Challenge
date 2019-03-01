import json
import random
from typing import Dict, List, Any

import spacy
import torch
import torch.nn.functional as F

from textworld import EnvInfos

from agents.utils.generic import to_np, to_pt, preproc, _words_to_ids, pad_sequences, max_len
from agents.utils.eps_scheduler import LinearScheduler
from agents.baseline_dqn.model import LSTM_DQN


class CustomAgent:
    def __init__(self):
        self.word2id = {}
        self.word_vocab = []
        self._load_vocab(vocab_file="./vocab.txt")
        self.EOS_id = self.word2id["</S>"]
        self.SEP_id = self.word2id["SEP"]

        with open("agents/baseline_dqn/config.json", "r") as fp:
            config = json.load(fp)

        self.use_cuda = True and torch.cuda.is_available()
        self.model = LSTM_DQN(config=config["model"], word_vocab=self.word_vocab)
        self.action_head = None

        self.nlp = spacy.load('en', disable=['ner', 'parser', 'tagger'])

        self.eps_scheduler = LinearScheduler({})
        self.act_steps = 0

        self._episode_started = False

    def act(self, obs: List[str], scores: List[int], dones: List[bool], infos: Dict[str, List[Any]]) -> List[str]:
        if not self._episode_started:
            self.init(obs, infos)

        # Get infos
        admissible_commands = infos["admissible_commands"]
        if random.random() < self.eps_scheduler(self.act_steps):
            return random.choice(admissible_commands)
        else:
            pass

    def get_game_state_info(self, obs: Dict[str], infos: [Dict[str, List[Any]]]):
        inventory_token_list = [preproc(item, tokenizer=self.nlp) for item in infos["inventory"]]
        inventory_id_list = [_words_to_ids(tokens, self.word2id) for tokens in inventory_token_list]

        feedback_token_list = [preproc(item, str_type="feedback", tokenizer=self.nlp) for item in infos["inventory"]]
        feedback_id_list = [_words_to_ids(tokens, self.word2id) for tokens in feedback_token_list]

        recipe_token_list = [preproc(item, tokenizer=self.nlp) for item in infos["extra.recipe"]]
        recipe_id_list = [_words_to_ids(tokens, self.word2id) for tokens in recipe_token_list]

        description_token_list = [preproc(item, tokenizer=self.nlp) for item in infos["description"]]
        for i, d in enumerate(description_token_list):
            if len(d) == 0:
                description_token_list[i] = ["end"]
        description_id_list = [_words_to_ids(tokens, self.word2id) for tokens in description_token_list]

        state_id_list = [d + [self.SEP_id] +
                         i + [self.SEP_id] + r
                         for (d, i, r) in zip(description_id_list, inventory_id_list, recipe_id_list)]
        pad_sequences(state_id_list, max_len=max_len(state_id_list))

    def init(self, obs: List[str], infos: Dict[str, List[Any]]) -> None:
        self._episode_started = True

    def _load_vocab(self, vocab_file: str) -> None:
        with open(vocab_file, "r") as fp:
            self.word_vocab = fp.read().split("\n")
        for i, w in enumerate(self.word_vocab + ['SEP']):
            self.word2id[w] = i
