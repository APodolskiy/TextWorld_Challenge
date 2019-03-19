from numpy import random
from collections import namedtuple, defaultdict
from multiprocessing import Queue
from typing import List, Dict, Any, Optional, Tuple

from textworld import EnvInfos

from agents.multiprocessing_agent.bert_net import QNet
from agents.multiprocessing_agent.utils import clean_text
from agents.utils.params import Params

State = namedtuple("State", ("description", "feedback", "inventory"))

Transition = namedtuple(
    "Transition",
    (
        "previous_state",
        "next_state",
        "action",
        "reward",
        "exploration_bonus",
        "done",
        "allowed_actions",
    ),
)


def idx_select(collection: List, indices: List, reversed_indices=False) -> List:
    """
    performs fancy indexing
    """
    if not indices:
        return []
    if isinstance(indices[0], bool):
        if reversed_indices:
            indices = [not idx for idx in indices]
        return [collection[i] for i, idx in enumerate(indices) if idx]
    return [collection[idx] for idx in indices]


class BaseQlearningAgent:
    """ Q-learning agent that requires all available information and therefore receives maximum
    penalty
    """

    def __init__(
        self,
        params: Params,
        net: QNet,
        eps_scheduler
    ) -> None:
        self._initialized = False
        self._episode_has_started = False
        self.max_steps_per_episode = params.pop("max_steps_per_episode")
        self.batch_size = params.get("n_parallel_envs")
        self.net = net
        self.eps_scheduler = eps_scheduler
        self.current_step = 0
        self.history = defaultdict(list)
        self.gamefile = None
        self.visited_states = defaultdict(set)
        self.training = False
        self.prev_actions = None
        self.prev_states = None
        self.prev_not_done_idxs = None

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
        self.prev_not_done_idxs = None
        self.visited_states = defaultdict(set)
        self.history = defaultdict(list)

    def reset(self):
        return self._end_episode()

    def act(
        self,
        observations: List[str],
        rewards: List[int],
        dones: List[bool],
        infos: Dict[str, List[Any]],
    ):
        self.gamefile = infos["gamefile"]
        actions = ["pass" for _ in range(len(observations))]
        not_done_idxs = idx_select(
            list(range(self.batch_size)), dones, reversed_indices=True
        )
        batch_admissible_commands = infos["admissible_commands"]
        commands_not_finished = idx_select(batch_admissible_commands, not_done_idxs)
        states = [
            State(
                description=clean_text(description, "description"),
                feedback=clean_text(obs, "feedback"),
                inventory=clean_text(inventory, "inventory"),
            )
            for description, obs, inventory in zip(
                infos["description"], observations, infos["inventory"]
            )
        ]

        # TODO: hzhz
        if random.random() < self.eps_scheduler.eps:
            selected_action_idxs = [
                random.choice(len(adm_com)) for adm_com in commands_not_finished
            ]
        else:
            self.net.eval()
            if not_done_idxs:
                q_values = self.net(
                    idx_select(states, not_done_idxs), commands_not_finished
                )
                selected_action_idxs = [q_val.argmax().item() for q_val in q_values]
            else:
                selected_action_idxs = []

        # self.net.eval()
        # q_values = self.net(states, batch_admissible_commands)
        # selected_action_idxs = [softmax(q_val, dim=1).multinomial(1).item() for q_val in q_values]
        for not_done_idx, adm_com, sel_act_idx in zip(
            not_done_idxs, commands_not_finished, selected_action_idxs
        ):
            actions[not_done_idx] = adm_com[sel_act_idx]
        self.update_experience_replay_buffer(
            not_done_idxs=not_done_idxs,
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
        self,
        not_done_idxs,
        next_states,
        actions,
        batch_admissible_commands,
        rewards,
        dones,
        is_lost,
    ):
        if self.prev_actions:
            idx = 0
            for (
                previous_state,
                next_state,
                action,
                admissible_commands,
                reward,
                done,
                game_lost,
            ) in zip(
                idx_select(self.prev_states, self.prev_not_done_idxs),
                idx_select(next_states, self.prev_not_done_idxs),
                idx_select(self.prev_actions, self.prev_not_done_idxs),
                idx_select(batch_admissible_commands, self.prev_not_done_idxs),
                idx_select(rewards, self.prev_not_done_idxs),
                idx_select(dones, self.prev_not_done_idxs),
                idx_select(is_lost, self.prev_not_done_idxs),
            ):
                assert action != "pass"
                desc_inventory = next_state.description + next_state.inventory
                reward, exploration_bonus = self.calculate_rewards(
                    reward=reward, game_lost=game_lost, done=done, state=desc_inventory
                )
                self.visited_states[self.gamefile].add(desc_inventory)
                self.history[self.prev_not_done_idxs[idx]].append(
                    Transition(
                        previous_state=previous_state,
                        next_state=next_state,
                        action=action,
                        allowed_actions=admissible_commands,
                        exploration_bonus=exploration_bonus,
                        reward=reward,
                        done=done,
                    )
                )
                idx += 1

        self.prev_actions = actions
        self.prev_states = next_states
        self.prev_not_done_idxs = not_done_idxs

    def calculate_rewards(self, reward: int, game_lost: bool, done: bool, state: str):
        reward = float(reward)
        exploration_bonus = float(state not in self.visited_states[self.gamefile])
        if done:
            if game_lost:
                reward = -2.0
                exploration_bonus = 0.0
            else:
                reward = 2.0
        return reward, exploration_bonus
