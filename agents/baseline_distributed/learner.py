import time
from typing import Dict

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from agents.baseline_distributed.model import LSTM_DQN
from agents.baseline_distributed.utils.preprocess import SEP_TOKEN, ITM_TOKEN
from agents.baseline_distributed.utils.utils import SimpleTransition
from agents.utils.generic import pad_sequences, max_len, to_pt
from agents.utils.replay import TernaryPrioritizeReplayMemory


class Learner:
    def __init__(self, config: Dict, shared_state, shared_memory: TernaryPrioritizeReplayMemory):
        self.word_vocab = []
        self.word2id = {}
        self._load_vocab(vocab_file="./vocab.txt")
        self.EOS_id = self.word2id["</s>"]
        self.SEP_id = self.word2id[SEP_TOKEN]
        self.ITM_id = self.word2id[ITM_TOKEN]

        self.config = config
        self.replay_batch_size = self.config["training"]["replay_batch_size"]
        self.clip_grad_norm = self.config["training"]["clip_grad_norm"]
        self.discount_gamma = self.config["training"]["discount_gamma"]
        self.update_per_k_game_steps = self.config["training"]["update_freq"]
        self.update_target = self.config["training"]["target_net_update_freq"]

        self.model = LSTM_DQN(config=self.config["model"],
                              word_vocab_size=len(self.word_vocab) + 2)
        self.target_model = LSTM_DQN(config=self.config["model"],
                                     word_vocab_size=len(self.word_vocab) + 2)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.replay_memory = shared_memory
        self.shared_state = shared_state

        self.shared_state['model'] = {k: v.cpu() for k, v in self.model.state_dict().items()}

    def update(self) -> float:
        transitions = self.replay_memory.sample(self.replay_batch_size)
        batch = SimpleTransition(*zip(*transitions))

        description_id = pad_sequences(batch.description_ids,
                                       maxlen=max_len(batch.description_ids)).astype('int32')
        input_description = to_pt(description_id, self.use_cuda)
        preprocessed_commands = [pad_sequences(commands, maxlen=max_len(commands)).astype('int32')
                                 for commands in batch.command_ids]
        # preprocessed_commands = pad_sequences(batch.command_ids,
        #                                       maxlen=max_len(batch.command_ids)).astype('int32')
        input_commands = [to_pt(commands, self.use_cuda) for commands in preprocessed_commands]
        #input_commands = to_pt(preprocessed_commands, self.use_cuda)
        q_values = self.model(input_description, input_commands)
        q_values = torch.stack(q_values, dim=0).squeeze(1)

        next_description = pad_sequences(batch.next_description_ids,
                                         maxlen=max_len(batch.next_description_ids))
        next_input_description = to_pt(next_description, self.use_cuda)

        next_preprocessed_commands = [pad_sequences(commands, maxlen=max_len(commands))
                                      for commands in batch.next_command_ids]
        next_input_commands = [to_pt(commands, self.use_cuda) for commands in next_preprocessed_commands]
        next_q_values_target = self.target_model(next_input_description, next_input_commands)
        next_q_values_target = [q_value.detach() for q_value in next_q_values_target]
        next_q_values_model = self.model(next_input_description, next_input_commands)
        next_q_values_model = [q_value.detach() for q_value in next_q_values_model]
        next_q_value = [target_q_values[q_vals.argmax()]
                        for q_vals, target_q_values in zip(next_q_values_model, next_q_values_target)]
        next_q_value = torch.stack(next_q_value, dim=0)

        rewards = to_pt(np.array(batch.reward, dtype='float32'), self.use_cuda, type='float')
        not_done = 1.0 - np.array(batch.done, dtype='float32')
        not_done = to_pt(not_done, self.use_cuda, type='float')
        rewards = rewards + not_done * next_q_value * self.discount_gamma
        loss = F.smooth_l1_loss(q_values, rewards)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        return loss.item()

    def learn(self, max_steps: int = 5_000_000) -> None:
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.target_model.to(self.device)
        while self.replay_memory.get_size() < self.replay_batch_size:
            time.sleep(1)

        for i in range(max_steps):
            loss = self.update()
            print(f"Current loss {loss}")
            if i % self.update_per_k_game_steps == 0:
                self.shared_state['model'] = {k: v for k, v in self.model.state_dict().items()}

            if i % self.update_target == 0:
                self.target_model.load_state_dict(self.model.state_dict())

    def _load_vocab(self, vocab_file: str) -> None:
        with open(vocab_file, "r") as fp:
            self.word_vocab = fp.read().split("\n")
        for i, w in enumerate(self.word_vocab + [SEP_TOKEN] + [ITM_TOKEN]):
            self.word2id[w] = i
