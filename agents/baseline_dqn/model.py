import logging
from typing import Dict
from types import SimpleNamespace

import torch.nn as nn
import torch.nn.functional as F


class ModelConfig:
    def __init__(self, config: Dict):
        self.embed_size = config["embedding_size"]
        self.h_size = config["rnn_hidden_size"]
        self.h_dropout = config["dropout_between_rnn_layers"]


class LSTM_DQN(nn.Module):
    def __init__(self, config, word_vocab):
        super(LSTM_DQN, self).__init__()
        self.config = ModelConfig(config)

        self.word_vocab_size = len(word_vocab)
        self.id2word = word_vocab
        self.embedding = nn.Embedding(self.config.embed_size, self.word_vocab_size)
        self.state_encoder = nn.LSTM(input_size=self.config.embed_size,
                                     hidden_size=self.config.h_size)
        self.action_encoder = nn.LSTM(input_size=self.config.embed_size,
                                      hidden_size=self.config.h_size)

    def forward(self, infos: Dict[str]):
        pass
