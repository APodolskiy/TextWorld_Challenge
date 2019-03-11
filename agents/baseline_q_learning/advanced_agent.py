import random
from typing import List, Dict, Any, Optional

import torch
from pytorch_pretrained_bert import BertModel, BertTokenizer
from textworld import EnvInfos
from torch.nn import Module
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from agents.utils.eps_scheduler import EpsScheduler
from agents.utils.params import Params


class QNet(Module):
    def __init__(self, config: Params):
        super().__init__()

        self.hidden_size = config.pop("hidden_size")
        self.embedding_size = config.pop("embedding_size")

        self.obs_to_hidden = nn.Linear(self.embedding_size, self.hidden_size)
        self.actions_to_hidden = nn.Linear(self.embedding_size, self.hidden_size)
        self.hidden_to_scores = nn.Linear(self.hidden_size, 1)

        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, observation, actions):
        obs = self.obs_to_hidden(observation)
        actions = self.actions_to_hidden(actions)
        final_state = self.lrelu(obs * actions)
        q_values = self.hidden_to_scores(final_state)
        return q_values


class BaseQlearningAgent:
    """ Q-learning agent that requires all available information and therefore receives maximum
    penalty
    """

    def __init__(self, config: Params) -> None:
        self._initialized = False
        self._epsiode_has_started = False
        self.device = config.pop("device")
        self.max_steps_per_episode = config.pop("max_steps_per_episode")

        self.bert = (
            BertModel.from_pretrained("bert-base-uncased").to(self.device).eval()
        )
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.qnet = QNet(config.pop("network"))
        self.eps_scheduler = EpsScheduler(config.pop("epsilon"))

        self.current_step = 0

    def train(self) -> None:
        """ Tell the agent it is in training mode. """
        pass  # [You can insert code here.]

    def eval(self) -> None:
        """ Tell the agent it is in evaluation mode. """
        pass  # [You can insert code here.]

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

        self._epsiode_has_started = True

        # [You can insert code here.]

    def _end_episode(
        self, obs: List[str], scores: List[int], infos: Dict[str, List[Any]]
    ) -> None:
        """
        Tell the agent the episode has terminated.

        Arguments:
            obs: Previous command's feedback for each game.
            score: The score obtained so far for each game.
            infos: Additional information for each game.
        """
        self._epsiode_has_started = False

        # [You can insert code here.]

    def act(
        self,
        obs: str,
        scores: List[int],
        dones: List[bool],
        infos: Dict[str, List[Any]],
    ):
        """
        Acts upon the current list of observations.

        One text command must be returned for each observation.

        Arguments:
            obs: Previous command's feedback for each game.
            scores: The score obtained so far for each game.
            dones: Whether a game is finished.
            infos: Additional information for each game.

        Returns:
            Text commands to be performed (one per observation).
            If episode had ended (e.g. `all(dones)`), the returned
            value is ignored.

        Notes:
            Commands returned for games marked as `done` have no effect.
            The states for finished games are simply copy over until all
            games are done.
        """
        # if all(dones):
        #     self._end_episode(obs, scores, infos)
        #     return  # Nothing to return.
        #
        # if not self._epsiode_has_started:
        #     self._start_episode(obs, infos)
        # TODO: no batching currecntly
        admissible_commands = infos["admissible_commands"]

        if random.random() < self.eps_scheduler.eps(self.current_step):
            command = random.choice(admissible_commands)
        else:
            state_repr = self.embed_observation(obs, infos)
            actions_repr = self.embed_actions(admissible_commands)
            q_values = self.qnet(state_repr, torch.cat(actions_repr, dim=0))
            max_q_val, idx_max_q_val = q_values.max()
            command = admissible_commands[idx_max_q_val]
        self.current_step += 1
        return command
        # raise NotImplementedError()
        # [Insert your code here to obtain the commands.]
        # return ["wait"] * len(obs)  # No-op

    def embed_observation(self, obs: str, infos: Dict):
        # TODO: change sep to smth other?
        state_description = (
            "[CLS] "
            + "[SEP]".join([obs, infos["description"], infos["inventory"]])
            + " [SEP]"
        )

        tokenized_state_description = self.tokenizer.tokenize(state_description)
        cleaned_tokenized_state_decription = [
            token
            for token in tokenized_state_description
            if token not in {"$", "|", "", "_", "\\", "/"}
        ]
        indexed_state_description = self.tokenizer.convert_tokens_to_ids(
            cleaned_tokenized_state_decription
        )
        indexed_state_description = torch.tensor(
            [indexed_state_description], device=self.device
        )
        # TODO: fine-tune?
        with torch.no_grad():
            _, state_repr = self.bert(indexed_state_description)
        return state_repr

    def embed_actions(self, actions):
        embedded_actions = []
        with torch.no_grad():
            for action in actions:
                tokenzed_action = self.tokenizer.tokenize(f"[CLS] {action} [SEP]")
                action_indices = torch.tensor(
                    [self.tokenizer.convert_tokens_to_ids(tokenzed_action)],
                    device=self.device,
                )
                _, action_embedding = self.bert(action_indices)
                embedded_actions.append(action_embedding)
        return embedded_actions
