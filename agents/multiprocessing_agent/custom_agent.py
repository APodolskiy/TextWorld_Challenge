import random
from collections import namedtuple
from multiprocessing import Queue
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer
from textworld import EnvInfos
from torch.nn import Module, LayerNorm
from torch.nn.utils.rnn import pad_sequence

from agents.utils.eps_scheduler import EpsScheduler
from agents.utils.params import Params

State = namedtuple("State", ("description", "feedback", "inventory"))

Transition = namedtuple(
    "Transition",
    ("previous_state", "next_state", "action", "reward", "done", "allowed_actions"),
)


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


class BaseQlearningAgent:
    """ Q-learning agent that requires all available information and therefore receives maximum
    penalty
    """

    def __init__(
        self,
        params: Params,
        net: QNet,
        eps_scheduler_params,
        experience_replay_buffer: Optional[Queue] = None,
    ) -> None:
        self._initialized = False
        self._episode_has_started = False
        self.max_steps_per_episode = params.pop("max_steps_per_episode")
        self.experience_replay_buffer = experience_replay_buffer
        self.net = net
        self.eps_scheduler = EpsScheduler(eps_scheduler_params)
        self.current_step = 0
        self.training = False
        self.prev_actions = None
        self.prev_states = None
        self.already_dones = None

    def train(self) -> None:
        """ Tell the agent it is in training mode. """
        self.training = True

    def eval(self) -> None:
        """ Tell the agent it is in evaluation mode. """
        self.training = False

    def select_additional_infos(self) -> EnvInfos:
        """
        Returns what additional information should be made available at each game step.

        Requested information will be included within the `infos` dictionary
        passed to `CustomAgent.act()`. To request specific information, create a
        :py:class:`textworld.EnvInfos <textworld.envs.wrappers.filter.EnvInfos>`
        and set the appropriate attributes to `True`. The possible choices are:

        * `description`: text description of the current room, i.e. output of the `look` command;
        * `inventory`: text listing of the player's inventory, i.e. output of the `inventory` command;
        * `max_score`: maximum reachable score of the game;
        * `objective`: objective of the game described in text;
        * `entities`: names of all entities in the game;
        * `verbs`: verbs understood by the the game;
        * `command_templates`: templates for commands understood by the the game;
        * `admissible_commands`: all commands relevant to the current state;

        In addition to the standard information, game specific information
        can be requested by appending corresponding strings to the `extras`
        attribute. For this competition, the possible extras are:

        * `'recipe'`: description of the cookbook;
        * `'walkthrough'`: one possible solution to the game (not guaranteed to be optimal);

        Example:
            Here is an example of how to request information and retrieve it.

            >>> from textworld import EnvInfos
            >>> request_infos = EnvInfos(description=True, inventory=True, extras=["recipe"])
            ...
            >>> env = gym.make(env_id)
            >>> ob, infos = env.reset()
            >>> print(infos["description"])
            >>> print(infos["inventory"])
            >>> print(infos["extra.recipe"])

        Notes:
            The following information *won't* be available at test time:

            * 'walkthrough'

            Requesting additional infos comes with some penalty (called handicap).
            The exact penalty values will be defined in function of the average
            scores achieved by agents using the same handicap.

            Handicap is defined as follows
                max_score, has_won, has_lost,               # Handicap 0
                description, inventory, verbs, objective,   # Handicap 1
                command_templates,                          # Handicap 2
                entities,                                   # Handicap 3
                extras=["recipe"],                          # Handicap 4
                admissible_commands,                        # Handicap 5
        """
        return EnvInfos(
            description=True,
            inventory=True,
            extras=["recipe"],
            admissible_commands=True,
        )

    def _init(self) -> None:
        """ Initialize the agent. """
        self._initialized = True

        # [You can insert code here.]

    def _start_episode(self, obs: List[str], infos: Dict[str, List[Any]]) -> None:
        """
        Prepare the agent for the upcoming episode.

        Arguments:
            obs: Initial feedback for each game.
            infos: Additional information for each game.
        """
        if not self._initialized:
            self._init()

        self._episode_has_started = True

        # [You can insert code here.]

    def _end_episode(self) -> None:
        """
        Tell the agent the episode has terminated.

        Arguments:
            obs: Previous command's feedback for each game.
            score: The score obtained so far for each game.
            infos: Additional information for each game.
        """
        self._episode_has_started = False
        self.prev_actions = None
        self.prev_states = None
        self.already_dones = None
        # [You can insert code here.]

    def reset(self):
        return self._end_episode()

    def act(
        self,
        observations: List[str],
        rewards: List[int],
        dones: List[bool],
        infos: Dict[str, List[Any]],
    ):
        batch_admissible_commands = infos["admissible_commands"]
        states = [
            State(description=description, feedback=obs, inventory=inventory)
            for description, obs, inventory in zip(
                infos["description"], observations, infos["inventory"]
            )
        ]
        # TODO: hzhz
        if random.random() < self.eps_scheduler.eps:
            actions = [random.choice(adm_com) for adm_com in batch_admissible_commands]
        else:
            self.net.eval()
            q_values = self.net(states, batch_admissible_commands)
            selected_idxs = [q_val.argmax().item() for q_val in q_values]
            actions = [
                acts[idxs]
                for acts, idxs in zip(batch_admissible_commands, selected_idxs)
            ]
        self.update_experience_replay_buffer(
            next_states=states,
            actions=actions,
            batch_admissible_commands=batch_admissible_commands,
            rewards=rewards,
            dones=dones,
            is_lost=infos["is_lost"],
        )
        self.current_step += 1
        return actions

    def update_experience_replay_buffer(
        self, next_states, actions, batch_admissible_commands, rewards, dones, is_lost
    ):
        if self.prev_actions:
            for (
                previous_state,
                next_state,
                action,
                admissible_commands,
                reward,
                done,
                already_done,
                game_lost,
            ) in zip(
                self.prev_states,
                next_states,
                self.prev_actions,
                batch_admissible_commands,
                rewards,
                dones,
                self.already_dones,
                is_lost,
            ):
                if not already_done:
                    reward = float(reward)
                    if done:
                        if game_lost:
                            reward = -1.0
                        else:
                            reward = 2.0
                    self.experience_replay_buffer.put(
                        Transition(
                            previous_state=previous_state,
                            next_state=next_state,
                            action=action,
                            allowed_actions=admissible_commands,
                            reward=reward,
                            done=done,
                        )
                    )

        self.prev_actions = actions
        self.prev_states = next_states
        self.already_dones = dones
