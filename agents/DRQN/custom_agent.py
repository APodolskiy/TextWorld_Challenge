from numpy import random
from collections import namedtuple, defaultdict
from typing import List, Dict, Any

from textworld import EnvInfos

from agents.DRQN.utils import clean_text, idx_select
from agents.utils.params import Params

State = namedtuple("State", ("description", "feedback", "inventory", "prev_action"))

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


class BaseQlearningAgent:
    """ Q-learning agent that requires all available information and therefore receives maximum
    penalty
    """

    def __init__(self, params: Params, net, eps_scheduler) -> None:
        self._initialized = False
        self.max_steps_per_episode = params.pop("max_steps_per_episode")
        self.batch_size = params.get("n_parallel_envs")
        self.net = net
        self.eps_scheduler = eps_scheduler
        self.reset()
        self._episode_has_started = True

    def train(self) -> None:
        """ Tell the agent it is in training mode. """
        self.training = True

    def eval(self) -> None:
        """ Tell the agent it is in evaluation mode. """
        self.training = False

    @staticmethod
    def select_additional_infos() -> EnvInfos:
        """
        Returns what additional information should be made available at each game step.

        """
        return EnvInfos(
            max_score=True,
            description=True,
            inventory=True,
            extras=["walkthrough", "recipe"],
            admissible_commands=True,
        )

    def _init(self) -> None:
        """ Initialize the agent. """
        self._initialized = True

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
        self.current_step = 0
        self.gamefile = None
        self._episode_has_started = False
        self.prev_actions = ["none" for _ in range(self.batch_size)]
        self.max_reward = None
        self.prev_cum_rewards = [0 for _ in range(self.batch_size)]
        self.prev_states = None
        self.prev_not_done_idxs = None
        self.visited_states = defaultdict(set)
        self.history = defaultdict(list)

    def reset(self):
        return self._end_episode()

    def act(
        self,
        observations: List[str],
        cum_rewards: List[int],
        dones: List[bool],
        infos: Dict[str, List[Any]],
    ):
        self.gamefile = infos.get("gamefile")
        infos["is_lost"] = [
            ("You lost!" in o if d else False) for o, d in zip(observations, dones)
        ]
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
                prev_action=action,
            )
            for description, obs, inventory, action in zip(
                infos["description"],
                observations,
                infos["inventory"],
                self.prev_actions,
            )
        ]

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
        self.max_reward = infos["max_score"][0]
        self.update_experience_replay_buffer(
            not_done_idxs=not_done_idxs,
            next_states=states,
            actions=actions,
            batch_admissible_commands=batch_admissible_commands,
            cum_rewards=cum_rewards,
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
        cum_rewards,
        dones,
        is_lost,
    ):
        if self.prev_states:
            idx = 0
            for (
                previous_state,
                next_state,
                action,
                admissible_commands,
                cum_reward,
                prev_cum_reward,
                done,
                game_lost,
            ) in zip(
                idx_select(self.prev_states, self.prev_not_done_idxs),
                idx_select(next_states, self.prev_not_done_idxs),
                idx_select(self.prev_actions, self.prev_not_done_idxs),
                idx_select(batch_admissible_commands, self.prev_not_done_idxs),
                idx_select(cum_rewards, self.prev_not_done_idxs),
                idx_select(self.prev_cum_rewards, self.prev_not_done_idxs),
                idx_select(dones, self.prev_not_done_idxs),
                idx_select(is_lost, self.prev_not_done_idxs),
            ):
                desc_inventory = next_state.description + next_state.inventory
                if self.current_step == 1:
                    self.visited_states[self.gamefile].add(
                        previous_state.description + previous_state.inventory
                    )
                reward, exploration_bonus = self.calculate_rewards(
                    cum_reward=cum_reward,
                    prev_cum_reward=prev_cum_reward,
                    game_lost=game_lost,
                    done=done,
                    state=desc_inventory,
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
        self.prev_states = next_states
        self.prev_actions = actions
        self.prev_not_done_idxs = not_done_idxs
        self.prev_cum_rewards = cum_rewards

    def calculate_rewards(
        self,
        cum_reward: int,
        prev_cum_reward: int,
        game_lost: bool,
        done: bool,
        state: str,
    ):
        reward = float(cum_reward - prev_cum_reward)
        exploration_bonus = 0.5 * float(state not in self.visited_states[self.gamefile])
        if game_lost:
            reward = -1.5
            exploration_bonus = 0.0
        if done and cum_reward == self.max_reward:
            reward = 2.0
            exploration_bonus = 0.0
        return reward, exploration_bonus
