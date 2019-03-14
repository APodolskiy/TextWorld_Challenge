import logging
from typing import Dict, List
from types import SimpleNamespace

import torch
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

        self.word_vocab_size = len(word_vocab) + 1
        self.id2word = word_vocab
        self.embedding = nn.Embedding(self.word_vocab_size, self.config.embed_size)
        self.state_encoder = nn.LSTM(input_size=self.config.embed_size,
                                     hidden_size=self.config.h_size)
        self.action_encoder = nn.LSTM(input_size=self.config.embed_size,
                                      hidden_size=self.config.h_size)

    def forward(self, description: List[int], commands: List[List[int]]):
        commands = commands.t()
        batch_size = commands.size(1)
        description_emb = self.embedding(description)
        commands_emb = self.embedding(commands)
        _, (s_h, s_c) = self.state_encoder(description_emb.unsqueeze(1))
        _, (a_h, a_c) = self.action_encoder(commands_emb,
                                            (s_h.repeat(1, batch_size, 1),
                                             s_c.repeat(1, batch_size, 1))
                                            )
        q_values = (s_h * a_h).sum(dim=-1)
        return q_values
