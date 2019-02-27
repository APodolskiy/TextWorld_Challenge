from typing import Dict, List, Any

import spacy
import torch
import torch.nn.functional as F

from textworld import EnvInfos

from agents.utils.generic import to_np, to_pt, preproc, _words_to_ids, pad_sequences, max_len


class CustomAgent:
    def __init__(self):
        self.word2id = {}
        self.word_vocab = []
        self._load_vocab(vocab_file="./vocab.txt")
        self.EOS_id = self.word2id["</S>"]

        self.use_cuda = True and torch.cuda.is_available()
        self.model = None
        self.action_head = None

        self.nlp = spacy.load('en', disable=['ner', 'parser', 'tagger'])

        self._episode_started = False

    def act(self, obs: List[str], scores: List[int], dones: List[bool], infos: Dict[str, List[Any]]) -> List[str]:
        if not self._episode_started:
            self.init(obs, infos)

        # Get infos
        admissible_commands = infos["admissible_commands"]
        pass

    def embed_obs_infos(self, obs: List[str], infos: Dict[str, List[Any]]):
        pass

    def embed_action(self, actions):
        pass

    def init(self, obs: List[str], infos: Dict[str, List[Any]]) -> None:
        self._episode_started = True

    def _load_vocab(self, vocab_file: str) -> None:
        with open(vocab_file, "r") as fp:
            self.word_vocab = fp.read().split("\n")
        for i, w in enumerate(self.word_vocab):
            self.word2id[w] = i
