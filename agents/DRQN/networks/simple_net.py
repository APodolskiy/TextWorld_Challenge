import pickle
from typing import List

import numpy
import spacy
import torch
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from torch import tensor
from torch.nn import Module, Embedding, Linear, ELU, LSTM, GRU
from torch.nn.functional import cosine_similarity, softmax
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from agents.utils.types import Transition


class SimpleNet(Module):
    def __init__(self, config, device, vocab_size, pad_idx=0):
        super().__init__()

        self.emb_dim = config.pop("embedding_dim")
        self.hidden_size = config.pop("hidden_size")
        self.d = numpy.sqrt(self.hidden_size)
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

        self.embedding = Embedding(
            self.vocab_size, self.emb_dim, padding_idx=self.pad_idx
        )

        self.state_embedder = PytorchSeq2SeqWrapper(
            LSTM(
                batch_first=True, input_size=self.emb_dim, hidden_size=self.hidden_size
            )
        )

        self.state_recurrence = PytorchSeq2VecWrapper(
            GRU(
                batch_first=True,
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
            )
        )

        self.action_embedder = PytorchSeq2VecWrapper(
            LSTM(
                batch_first=True, input_size=self.emb_dim, hidden_size=self.hidden_size
            )
        )

        self.recipe_embedder = PytorchSeq2VecWrapper(
            LSTM(
                batch_first=True, input_size=self.emb_dim, hidden_size=self.hidden_size
            )
        )

        self.state_to_hidden = Linear(self.hidden_size, self.hidden_size)
        self.state_to_hidden2 = Linear(self.hidden_size, self.hidden_size // 2)

        self.action_to_hidden = Linear(self.hidden_size, self.hidden_size)
        self.action_to_hidden2 = Linear(self.hidden_size, self.hidden_size // 2)

        self.elu = ELU()
        self.device = device

    def embed_states(self, states, recipe):
        state_mask = states != 0
        state_seq_embs = self.state_embedder(self.embedding(states), state_mask)
        att_weigths = (recipe * state_seq_embs / self.d).sum(dim=2)
        att_weigths[~state_mask] = float("-inf")
        att_probs = softmax(att_weigths, dim=1)
        return (att_probs.unsqueeze(2) * state_seq_embs).sum(dim=1)

    def forward(
        self,
        states: List[List[int]],
        actions: List[List[int]],
        recipe,
        hidden_states,
        mode,
    ):
        if not hidden_states is None:
            assert hidden_states.size(1) == len(states)

        recipe = torch.tensor(recipe, device=self.device)
        assert recipe.ndimension() == 1
        recipe = recipe.view(1, -1)
        recipe_emb = self.recipe_embedder(self.embedding(recipe), None).unsqueeze(1)

        assert mode in ["collect", "learn"]

        if mode == "collect":
            # states: [s1, ..., sn]
            state_batch = []
            for state in states:
                state_batch.append(torch.tensor(state, device=self.device))

            state_batch = pad_sequence(state_batch, batch_first=True)
            state_embs = self.embed_states(state_batch, recipe_emb)
            final_state_embs = self.state_recurrence(
                state_embs.unsqueeze(1), None, hidden_states
            )
            actions_batch = []
            for state_actions in actions:
                if isinstance(state_actions[0], int):
                    state_actions = [state_actions]
                actions_padded = pad_sequence(
                    [torch.tensor(a, device=self.device) for a in state_actions],
                    batch_first=True,
                )
                act_embs = self.action_embedder(
                    self.embedding(actions_padded), actions_padded != 0
                )
                actions_batch.append(act_embs)

            q_values = []
            for s, actions in zip(final_state_embs, actions_batch):
                s = s.unsqueeze(0)
                s_hidden = self.state_to_hidden(self.elu(s))
                act_hidden = self.action_to_hidden(self.elu(actions))

                q_values.append(3 * cosine_similarity(s_hidden, act_hidden, dim=1))

            return final_state_embs, q_values
        else:
            # states: [(s_1, s_2 ... s_k)_1, ... (s_1, s_2 ... s_k)_n]
            sequences = []
            for seq_states in states:
                if len(seq_states) < 8:
                    seq_states = [[0] for _ in range(8 - len(seq_states))] + seq_states
                sequences.append(
                    pad_sequence(
                        [torch.tensor(s, device=self.device) for s in seq_states],
                        batch_first=False,
                    )
                )
            state_batch = pad_sequence(sequences, batch_first=True)
            # batch_size X seq_len X state_len
            state_batch = state_batch.permute(0, 2, 1)

            embedded_state = []
            for seq_step in state_batch.split(1, dim=1):
                seq_step = seq_step.squeeze(1)
                state_embs = self.embed_states(seq_step, recipe_emb).unsqueeze(1)
                embedded_state.append(state_embs)
            state_embs = torch.cat(embedded_state, dim=1)
            final_state_embs = self.state_recurrence(
                state_embs, None, hidden_states
            )
            return

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


# class SimpleBowNet(Module):
#     def __init__(self, device, tokenizer):
#         super().__init__()
#         with open("vocab.txt") as f:
#             self.vocab = [line.rstrip("\n") for line in f]
#         self.token_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
#         self.vocab_size = len(self.vocab)
#         self.device = device
#         self.pad_idx = self.token_to_idx["<PAD>"]
#         self.unk_idx = self.token_to_idx["<UNK>"]
#         self.join_symbol = "<|>"
#         tok_exceptions = [
#             self.join_symbol,
#             [{ORTH: self.join_symbol, LEMMA: self.join_symbol, POS: "VERB"}],
#         ]
#         self.tokenizer = tokenizer
#         self.tokenizer.add_special_case(*tok_exceptions)
#         self.emb_dim = 150
#         self.hidden_size = 512
#         self.embedding = Embedding(
#             self.vocab_size, self.emb_dim, padding_idx=self.pad_idx
#         )
#         self.state_to_hidden = Linear(self.vocab_size, self.hidden_size // 4)
#         self.state_to_hidden2 = Linear(self.hidden_size, self.hidden_size // 2)
#
#         self.actions_to_hidden = Linear(self.vocab_size, self.hidden_size)
#         self.actions_to_hidden2 = Linear(self.hidden_size, self.hidden_size // 2)
#
#         self.hidden_to_hidden = Linear(self.hidden_size // 2, self.hidden_size // 4)
#         self.hidden_to_scores = Linear(self.hidden_size // 4, 1)
#         self.lrelu = LeakyReLU(0.2)
#
#     def vectorize(self, s: str):
#         raw_tokens = self.tokenizer(s.lower())
#         final_tokens = []
#         bad_symbols = {"_", "|", "\|"}
#         for token in raw_tokens:
#             if not token.is_space and not token.pos_ in ["PUNCT", "SYM"]:
#                 lemma = token.orth_.strip()
#                 if lemma and lemma not in bad_symbols and "$" not in lemma:
#                     final_tokens.append(lemma)
#         indices = [self.token_to_idx.get(t, self.unk_idx) for t in final_tokens]
#         return indices
#
#     def embed(self, idxs):
#         result = torch.zeros(
#             (len(idxs), self.vocab_size), dtype=torch.float32, device=self.device
#         )
#         for i, idx in enumerate(idxs):
#             for idx_ in idx:
#                 result[i][idx_] = 1
#         return result
#
#     def forward(self, states: List[State], actions: List[List[str]]):
#         state_batch = []
#         for state in states:
#             desc, obs, inventory = state.description, state.feedback, state.inventory
#             state_idxs = self.vectorize(
#                 f" {self.join_symbol} ".join([desc, obs, inventory])
#             )
#             state_batch.append(state_idxs)
#         state_batch = self.embed(state_batch)
#
#         actions_batch = []
#         for state_actions in actions:
#             if isinstance(state_actions, str):
#                 state_actions = [state_actions]
#             actions_batch.append(self.embed([self.vectorize(a) for a in state_actions]))
#         q_values = []
#         for state, actions in zip(state_batch, actions_batch):
#             hidden = self.state_to_hidden(state + actions)
#             # state = self.lrelu(self.state_to_hidden(state))
#             # state = self.lrelu(self.state_to_hidden2(state))
#             # actions = self.lrelu(self.actions_to_hidden(actions))
#             # actions = self.lrelu(self.actions_to_hidden2(actions))
#             # combined = state * actions
#             # hidden = self.lrelu(self.hidden_to_hidden(combined))
#             q_values.append(self.hidden_to_scores(hidden))
#
#         return q_values


if __name__ == "__main__":
    with open("transitions.pkl", "rb") as f:
        transitions: Transition = pickle.load(f)
    tokenizer = spacy.load("en_core_web_sm").tokenizer
    net = SimpleNet(device="cpu", tokenizer=tokenizer)
    net(transitions.previous_state[:4], transitions.allowed_actions[:4])
