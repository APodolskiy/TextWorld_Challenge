import pickle
from logging import warning

import torch
from typing import List

from spacy.attrs import LEMMA, ORTH, POS
from torch.nn import Module, Embedding, LSTM, Linear, GRU, LeakyReLU
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from agents.multiprocessing_agent.custom_agent import Transition, State
import spacy


class SimpleNet(Module):
    def __init__(self, device, tokenizer):
        super().__init__()
        with open("vocab.txt") as f:
            self.vocab = [line.rstrip("\n") for line in f]
        self.token_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.device = device

        self.pad_idx = self.token_to_idx["<PAD>"]
        self.unk_idx = self.token_to_idx["<UNK>"]
        self.join_symbol = "<|>"

        tok_exceptions = [
            self.join_symbol,
            [{ORTH: self.join_symbol, LEMMA: self.join_symbol, POS: "VERB"}],
        ]

        self.tokenizer = tokenizer
        self.tokenizer.add_special_case(*tok_exceptions)

        self.emb_dim = 300
        self.hidden_size = 1024

        self.embedding = Embedding(
            self.vocab_size, self.emb_dim, padding_idx=self.pad_idx
        )
        self.state_embedder = GRU(
            input_size=self.emb_dim, hidden_size=self.hidden_size, batch_first=True
        )
        self.action_embedder = GRU(
            batch_first=True, input_size=self.emb_dim, hidden_size=self.hidden_size
        )

        self.hidden_to_hidden = Linear(self.hidden_size, self.hidden_size)
        self.hidden_to_scores = Linear(self.hidden_size, 1)
        self.lrelu = LeakyReLU()

    def vectorize(self, s: str):
        raw_tokens = self.tokenizer(s.lower())
        final_tokens = []
        bad_symbols = {"_", "|", "\|"}
        for token in raw_tokens:
            if not token.is_space and not token.pos_ in ["PUNCT", "SYM"]:
                lemma = token.orth_.strip()
                if lemma and lemma not in bad_symbols and "$" not in lemma:
                    final_tokens.append(lemma)
        indices = [self.token_to_idx.get(t, self.unk_idx) for t in final_tokens]
        # TODO: remove
        # try:
        #     idx = indices.index(self.unk_idx)
        #     warning(f"Bad token: {final_tokens[idx]}")
        # except ValueError:
        #     pass
        return torch.tensor(indices, device=self.device)

    def embed(self, data_batch):
        batch = sorted(data_batch, key=len, reverse=True)
        lens = [len(s) for s in batch]
        batch = pad_sequence(batch, batch_first=True)
        embs = self.embedding(batch)
        packed_embs = pack_padded_sequence(embs, lens, batch_first=True)
        return packed_embs

    def forward(self, states: List[State], actions: List[List[str]]):
        state_batch = []
        for state in states:
            desc, obs, inventory = state.description, state.feedback, state.inventory
            state_idxs = self.vectorize(
                f" {self.join_symbol} ".join([desc, obs, inventory])
            )
            state_batch.append(state_idxs)
        state_embs = self.embed(state_batch)

        actions_batch = []
        for state_actions in actions:
            if isinstance(state_actions, str):
                state_actions = [state_actions]
            actions_batch.append(self.embed([self.vectorize(a) for a in state_actions]))

        _, state = self.state_embedder(state_embs)
        state = state.squeeze(0)

        q_values = []
        for s, actions in zip(state, actions_batch):
            _, act_state = self.action_embedder(actions)
            act_state = act_state.squeeze(0)
            combined = s * act_state
            hidden = self.lrelu(self.hidden_to_hidden(combined))
            q_values.append(self.hidden_to_scores(hidden))

        return q_values


class SimpleBowNet(Module):
    def __init__(self, device, tokenizer):
        super().__init__()
        with open("vocab.txt") as f:
            self.vocab = [line.rstrip("\n") for line in f]
        self.token_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.device = device
        self.pad_idx = self.token_to_idx["<PAD>"]
        self.unk_idx = self.token_to_idx["<UNK>"]
        self.join_symbol = "<|>"
        tok_exceptions = [
            self.join_symbol,
            [{ORTH: self.join_symbol, LEMMA: self.join_symbol, POS: "VERB"}],
        ]
        self.tokenizer = tokenizer
        self.tokenizer.add_special_case(*tok_exceptions)
        self.emb_dim = 300
        self.hidden_size = 1024
        self.embedding = Embedding(
            self.vocab_size, self.emb_dim, padding_idx=self.pad_idx
        )
        self.state_to_hidden = Linear(self.vocab_size, self.hidden_size)
        self.state_to_hidden2 = Linear(self.hidden_size, self.hidden_size // 2)

        self.actions_to_hidden = Linear(self.vocab_size, self.hidden_size)
        self.actions_to_hidden2 = Linear(self.hidden_size, self.hidden_size // 2)

        self.hidden_to_hidden = Linear(self.hidden_size // 2, self.hidden_size // 4)
        self.hidden_to_scores = Linear(self.hidden_size // 4, 1)
        self.lrelu = LeakyReLU(0.2)

    def vectorize(self, s: str):
        raw_tokens = self.tokenizer(s.lower())
        final_tokens = []
        bad_symbols = {"_", "|", "\|"}
        for token in raw_tokens:
            if not token.is_space and not token.pos_ in ["PUNCT", "SYM"]:
                lemma = token.orth_.strip()
                if lemma and lemma not in bad_symbols and "$" not in lemma:
                    final_tokens.append(lemma)
        indices = [self.token_to_idx.get(t, self.unk_idx) for t in final_tokens]
        return indices

    def embed(self, idxs):
        result = torch.zeros((len(idxs), self.vocab_size), dtype=torch.float32, device=self.device)
        for i, idx in enumerate(idxs):
            for idx_ in idx:
                result[i][idx_] = 1
        return result

    def forward(self, states: List[State], actions: List[List[str]]):
        state_batch = []
        for state in states:
            desc, obs, inventory = state.description, state.feedback, state.inventory
            state_idxs = self.vectorize(
                f" {self.join_symbol} ".join([desc, obs, inventory])
            )
            state_batch.append(state_idxs)
        state_batch = self.embed(state_batch)

        actions_batch = []
        for state_actions in actions:
            if isinstance(state_actions, str):
                state_actions = [state_actions]
            actions_batch.append(self.embed([self.vectorize(a) for a in state_actions]))
        q_values = []
        for state, actions in zip(state_batch, actions_batch):
            state = self.lrelu(self.state_to_hidden(state))
            state = self.lrelu(self.state_to_hidden2(state))
            actions = self.lrelu(self.actions_to_hidden(actions))
            actions = self.lrelu(self.actions_to_hidden2(actions))
            combined = state * actions
            hidden = self.lrelu(self.hidden_to_hidden(combined))
            q_values.append(self.hidden_to_scores(hidden))

        return q_values

if __name__ == "__main__":
    with open("transitions.pkl", "rb") as f:
        transitions: Transition = pickle.load(f)
    tokenizer = spacy.load("en_core_web_sm").tokenizer
    net = SimpleBowNet(device="cpu", tokenizer=tokenizer)
    net(transitions.previous_state[:4], transitions.allowed_actions[:4])
