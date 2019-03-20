import torch
from pytorch_pretrained_bert import BertModel, BertTokenizer
from torch import nn as nn
from torch.nn import Module, LayerNorm
from torch.nn.utils.rnn import pad_sequence

from agents.utils.params import Params


class QNet(Module):
    def __init__(self, config: Params, device="cuda"):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased").eval()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.hidden_size = config.get("hidden_size")
        self.embedding_size = config.get("embedding_size")

        # TODO: probably need recurrent nets here
        self.obs_to_hidden = nn.Linear(self.embedding_size, self.hidden_size)
        self.actions_to_hidden = nn.Linear(self.embedding_size, self.hidden_size)

        self.hidden_to_hidden = nn.Linear(self.hidden_size, self.hidden_size // 2)

        self.hidden_to_scores = nn.Linear(self.hidden_size // 2, 1)

        self.state_layer_norm = LayerNorm(self.hidden_size)
        self.action_layer_norm = LayerNorm(self.hidden_size)
        self.hidden_layer_norm = LayerNorm(self.hidden_size // 2)

        self.lrelu = nn.LeakyReLU(0.2)

        self.device = device

    # TODO: use receipt
    def forward(self, state_batch, actions_batch):
        observations, description, inventory = list(zip(*state_batch))
        # TODO: memory issues, how to deal?
        with torch.no_grad():
            embedded_observations = self.embed_observations(
                observations, description=description, inventory=inventory
            )
        q_values = []
        for obs, act in zip(embedded_observations, actions_batch):

            if isinstance(act, str):
                act = [act]
            with torch.no_grad():
                embedded_actions = self.embed_actions(act)
            obs = self.lrelu(self.obs_to_hidden(obs))
            actions = self.lrelu(self.actions_to_hidden(embedded_actions))
            final_state = self.state_layer_norm(obs) * self.action_layer_norm(actions)
            new_hidden_size = self.hidden_layer_norm(
                self.lrelu(self.hidden_to_hidden(final_state))
            )
            q_values.append(self.hidden_to_scores(new_hidden_size))
        return q_values

    def embed_observations(self, observations: str, description, inventory):
        obs_idxs = []
        for obs, descr, inventory in zip(observations, description, inventory):
            # TODO: change sep to smth other?
            state_description = (
                "[CLS] " + "[SEP]".join([obs, descr, inventory]) + " [SEP]"
            )

            tokenized_state_description = self.tokenizer.tokenize(state_description)
            cleaned_tokenized_state_decription = [
                token
                for token in tokenized_state_description
                if token not in {"$", "|", "", "_", "\\", "/"}
            ]

            # BERT does not support sentences longer than 512 tokens
            indexed_state_description = self.tokenizer.convert_tokens_to_ids(
                cleaned_tokenized_state_decription[:512]
            )
            indexed_state_description = torch.tensor(
                indexed_state_description, device=self.device
            )
            obs_idxs.append(indexed_state_description)
        padded_idxs = pad_sequence(obs_idxs, batch_first=True)
        _, state_repr = self.bert(padded_idxs)
        return state_repr

    def embed_actions(self, actions):
        embedded_actions = []
        # with torch.no_grad():
        for action in actions:
            tokenzed_action = self.tokenizer.tokenize(f"[CLS] {action} [SEP]")
            action_indices = torch.tensor(
                [self.tokenizer.convert_tokens_to_ids(tokenzed_action)],
                device=self.device,
            )
            _, action_embedding = self.bert(action_indices)
            embedded_actions.append(action_embedding)
        return torch.cat(embedded_actions)