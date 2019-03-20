import pickle
from typing import List

import numpy
import spacy
import torch
from spacy.attrs import LEMMA, ORTH, POS
from torch.nn import Module, Embedding, Linear, GRU, LeakyReLU
from torch.nn.functional import cosine_similarity
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from agents.DRQN.custom_agent import Transition, State


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

        self.emb_dim = 150
        self.hidden_size = 256

        self.embedding = Embedding(
            self.vocab_size, self.emb_dim, padding_idx=self.pad_idx
        )
        self.state_embedder = GRU(
            input_size=self.emb_dim, hidden_size=self.hidden_size, batch_first=True
        )

        self.action_embedder = GRU(
            batch_first=True, input_size=self.emb_dim, hidden_size=self.hidden_size
        )
        self.state_to_hidden = Linear(self.hidden_size, self.hidden_size)

        # self.state_memory = GRUCell(self.hidden_size, self.hidden_size)

        self.action_to_hidden = Linear(self.hidden_size, self.hidden_size)

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
        return torch.tensor(indices, device=self.device)

    def forward(self, states: List[State], actions: List[List[str]]):
        state_batch = []
        for state in states:
            desc, obs, inventory, prev_action = (
                state.description,
                state.feedback,
                state.inventory,
                state.prev_action,
            )
            state_idxs = self.vectorize(
                f" {self.join_symbol} ".join([desc, obs, inventory, state.prev_action])
            )
            state_batch.append(state_idxs)
        state_embs, orig_state_idxs = self.embed(state_batch)

        actions_batch = []
        actions_idxs_batch = []
        for state_actions in actions:
            if isinstance(state_actions, str):
                state_actions = [state_actions]
            act_embs, act_idxs = self.embed([self.vectorize(a) for a in state_actions])
            actions_batch.append(act_embs)
            actions_idxs_batch.append(act_idxs)

        seq_states, state = self.state_embedder(state_embs)

        # if hiddens is None:
        #     hiddens = [
        #         torch.zeros((1, self.hidden_size)).to(self.device)
        #         for _ in range(len(states))
        #     ]

        q_values = []
        state = state.squeeze(0)
        state = state[orig_state_idxs]
        for s, actions, act_idxs in zip(state, actions_batch, actions_idxs_batch):

            _, act_state = self.action_embedder(actions)
            act_state = act_state.squeeze(0)
            act_state = act_state[act_idxs]
            s_hidden = self.state_to_hidden(s)

            # s_hidden = self.state_memory(s_hidden.unsqueeze(0), hidden)
            # s_hidden = s_hidden.squeeze(0)

            act_hidden = self.action_to_hidden(act_state)

            q_values.append(
                3 * cosine_similarity(s_hidden.unsqueeze(0), act_hidden, dim=1)
            )

            # hidden = s_hidden

        return q_values

    def embed(self, data_batch):
        orig_lens = torch.tensor([len(s) for s in data_batch])
        sorted_idxs = torch.argsort(orig_lens, descending=True)
        orig_order = torch.argsort(sorted_idxs, descending=False)
        sorted_data_batch = [data_batch[idx] for idx in sorted_idxs]
        embs = self.embedding(pad_sequence(sorted_data_batch, batch_first=True))
        packed_embs = pack_padded_sequence(
            embs, [len(x) for x in sorted_data_batch], batch_first=True
        )
        assert all(
            [
                torch.equal(
                    numpy.array(sorted_data_batch, dtype=object)[orig_order.numpy()][i],
                    data_batch[i],
                )
                for i in range(len(data_batch))
            ]
        )
        return packed_embs, orig_order


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
        self.emb_dim = 150
        self.hidden_size = 512
        self.embedding = Embedding(
            self.vocab_size, self.emb_dim, padding_idx=self.pad_idx
        )
        self.state_to_hidden = Linear(self.vocab_size, self.hidden_size // 4)
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
        result = torch.zeros(
            (len(idxs), self.vocab_size), dtype=torch.float32, device=self.device
        )
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
            hidden = self.state_to_hidden(state + actions)
            # state = self.lrelu(self.state_to_hidden(state))
            # state = self.lrelu(self.state_to_hidden2(state))
            # actions = self.lrelu(self.actions_to_hidden(actions))
            # actions = self.lrelu(self.actions_to_hidden2(actions))
            # combined = state * actions
            # hidden = self.lrelu(self.hidden_to_hidden(combined))
            q_values.append(self.hidden_to_scores(hidden))

        return q_values


if __name__ == "__main__":
    with open("transitions.pkl", "rb") as f:
        transitions: Transition = pickle.load(f)
    tokenizer = spacy.load("en_core_web_sm").tokenizer
    net = SimpleNet(device="cpu", tokenizer=tokenizer)
    net(transitions.previous_state[:4], transitions.allowed_actions[:4])
