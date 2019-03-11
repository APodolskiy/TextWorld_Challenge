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


class BaseQlearningAgent:
    """ Q-learning agent that requires all available information and therefore receives maximum
    penalty
    """

    def __init__(self, config: Params) -> None:
        self._initialized = False
        self._epsiode_has_started = False
        self.device = config.pop("device")
        self.max_steps_per_episode = config.pop("max_steps_per_episode")

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
        obs: List[str],
        scores: List[int],
        dones: List[bool],
        infos: Dict[str, List[Any]],
    ):
        batch_admissible_commands = infos["admissible_commands"]
        command = [random.choice(adm_com) for adm_com in batch_admissible_commands]
        return command
