from typing import List

import spacy
import textworld.gym

import torch
import torch.multiprocessing as mp
import torch.optim as optim

from agents.baseline_distributed.model import LSTM_DQN
from agents.utils.eps_scheduler import LinearScheduler
from agents.utils.replay import TernaryPrioritizeReplayMemory


class Actor(mp.Process):
    def __init__(self, id: int, eps, request_infos, config, shared_state, shared_replay_memory, shared_writer):
        super(Actor, self).__init__()
        self.word_vocab = []
        self._load_vocab(vocab_file="./vocab.txt")
        self.EOS_id = self.word2id["</S>"]
        self.SEP_id = self.word2id["SEP"]

        self.writer = shared_writer

        self.request_infos = request_infos
        self.config = config
        self.nb_epochs = self.config['training']['nb_epochs']
        self.batch_size = self.config['training']['batch_size']
        self.max_nb_steps_per_episode = self.config['training']['max_nb_steps_per_episode']

        self.use_cuda = True and torch.cuda.is_available()
        self.model = LSTM_DQN(config=self.config["model"], word_vocab=self.word_vocab)

        self.nlp = spacy.load('en', disable=['ner', 'parser', 'tagger'])

        self.eps = eps
        self.eps_scheduler = LinearScheduler(self.config["exploration"])
        self.act_steps = 0
        self.episode_steps = 0
        self.num_episodes = 0

        self._episode_started = False
        self.previous_actions: List[str] = []
        self.scores: List[List[int]] = []
        self.dones: List[List[int]] = []
        self.prev_description_id: List = None
        self.prev_command: List = None

    def run(self, game_files):
        env_id = textworld.gym.register_games(game_files, self.request_infos, max_episode_steps=200, name='training')
        env_id = textworld.gym.make_batch(env_id, batch_size=1, parallel=False)

