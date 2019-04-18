import logging
from typing import Dict, List, Optional
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
    def __init__(self, config, word_vocab_size):
        super(LSTM_DQN, self).__init__()
        self.config = ModelConfig(config)

        self.word_vocab_size = word_vocab_size
        self.embedding = nn.Embedding(self.word_vocab_size, self.config.embed_size)
        self.state_encoder = nn.LSTM(input_size=self.config.embed_size,
                                     hidden_size=self.config.h_size, batch_first=True)
        self.action_encoder = nn.LSTM(input_size=self.config.embed_size,
                                      hidden_size=self.config.h_size, batch_first=True)
        self.state_to_hidden = nn.Linear(in_features=self.config.h_size,
                                         out_features=self.config.h_size)
        self.action_to_hidden = nn.Linear(in_features=self.config.h_size,
                                          out_features=self.config.h_size)
        self.hidden_to_hidden_act = nn.Linear(in_features=self.config.h_size,
                                              out_features=self.config.h_size)
        self.hidden_to_hidden_state = nn.Linear(in_features=self.config.h_size,
                                                out_features=self.config.h_size)
        self.nonlin = nn.ELU()

    def forward(self, description: List[int], commands: List[List[List[int]]], max_score: int = 3):
        # description embedding
        description_emb = self.embedding(description)
        _, (s_h, s_c) = self.state_encoder(description_emb)
        s_h = s_h.squeeze(0)
        s_c = s_c.squeeze(0)
        # state = self.hidden_to_hidden_state(self.nonlin(self.state_to_hidden(s_h)))
        state = self.state_to_hidden(s_h)
        # command embedding
        # TODO: add pack_padded_sequence
        q_values = []
        for idx, command_tokens in enumerate(commands):
            command_num = command_tokens.size(0)
            commands_emb = self.embedding(command_tokens)
            _, (a_h, a_c) = self.action_encoder(commands_emb  # ,
                                                # (s_h[idx].repeat(1, command_num, 1),
                                                # s_c[idx].repeat(1, command_num, 1))
                                                )

            # action = self.hidden_to_hidden_act(self.nonlin(self.action_to_hidden(a_h.squeeze(0))))
            action = self.action_to_hidden(a_h.squeeze(0))
            # env_q_values = (s_h[idx] * a_h).sum(dim=-1).squeeze(0)
            env_q_values = 3 * torch.cosine_similarity(state[idx].unsqueeze(0), action)
            # env_q_values = 3 * torch.cosine_similarity(state[idx].repeat(command_num, 1), a_h.squeeze(0))
            q_values.append(env_q_values)
        return q_values
