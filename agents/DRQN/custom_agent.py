import torch
from collections import defaultdict
from typing import List, Dict, Any

from numpy import random
from textworld import EnvInfos
from torch.nn.functional import softmax

from agents.utils.params import Params
from agents.utils.parsing import get_missing_ingredients_from_inventory, parse_inventory
from agents.utils.tokenization import SpacyVectorizer
from agents.utils.types import Transition, HistoryElement
from agents.utils.utils import clean_text, idx_select


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
        self.vectorizer = SpacyVectorizer()
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

    def start_episode(self, infos) -> None:
        """
        Prepare the agent for the upcoming episode.
        Arguments:
            infos: Additional information for each game.
        """
        from agents.utils.parsing import parse_recipe

        recipe_text = infos["extra.recipe"][0]
        ingredients, cooking_steps = parse_recipe(recipe_text)
        self.ingredients = ingredients
        self.cooking_steps = self.vectorizer(
            f" {self.vectorizer.join_symbol} ".join(cooking_steps)
        )

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
        self.ingredients = None
        self.cooking_steps = None
        self.hidden_state = None
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
        infos["feedback"] = observations
        infos["is_lost"] = [
            ("You lost!" in o if d else False) for o, d in zip(observations, dones)
        ]
        actions = ["pass" for _ in range(len(observations))]
        not_done_idxs = idx_select(
            list(range(self.batch_size)), dones, reversed_indices=True
        )
        batch_admissible_commands = [
            [self.vectorizer(c) for c in commands]
            for commands in infos["admissible_commands"]
        ]
        commands_not_finished = idx_select(batch_admissible_commands, not_done_idxs)
        states = [
            self.vectorize_state(
                feedback=obs,
                prev_action=prev_action,
                prev_cum_reward=prev_cum_reward,
                infos=infos,
                infos_index=idx,
            )
            for idx, (obs, prev_action, prev_cum_reward) in enumerate(
                zip(observations, self.prev_actions, self.prev_cum_rewards)
            )
        ]

        # if random.random() < self.eps_scheduler.eps:
        #     self.q_values = None
        #     selected_action_idxs = [
        #         random.choice(len(adm_com)) for adm_com in commands_not_finished
        #     ]
        # else:
        self.net.eval()
        if not_done_idxs:
            # TODO: update hidden state
            new_hidden_states, self.q_values = self.net(
                idx_select(states, not_done_idxs),
                commands_not_finished,
                self.cooking_steps,
                hidden_states=(
                    None
                    if self.hidden_state is None
                    else torch.stack(idx_select(self.hidden_state, not_done_idxs), dim=0).unsqueeze(0)
                ),
                mode="collect",
            )
            self.hidden_state = ["None" for _ in range(len(observations))]
            for idx, state in zip(not_done_idxs, new_hidden_states):
                self.hidden_state[idx] = state

            selected_action_idxs = [
                softmax(q_val / 0.1).multinomial(1).item() for q_val in self.q_values
            ]
            # selected_action_idxs = [
            #     q_val.argmax().item() for q_val in self.q_values
            # ]
        else:
            self.q_values = None
            selected_action_idxs = []

        for not_done_idx, adm_com, sel_act_idx in zip(
            not_done_idxs,
            idx_select(infos["admissible_commands"], not_done_idxs),
            selected_action_idxs,
        ):
            actions[not_done_idx] = adm_com[sel_act_idx]

        self.max_reward = infos["max_score"][0]
        self.update_history(
            not_done_idxs=not_done_idxs,
            next_states=states,
            actions=actions,
            batch_admissible_commands=batch_admissible_commands,
            cum_rewards=cum_rewards,
            dones=dones,
            infos=infos,
        )
        self.current_step += 1
        return actions

    def update_history(
        self,
        not_done_idxs,
        next_states,
        actions,
        batch_admissible_commands,
        cum_rewards,
        dones,
        infos,
    ):
        if self.prev_states is not None:
            for idx, not_done_idx in enumerate(self.prev_not_done_idxs):
                previous_state = self.prev_states[not_done_idx]
                next_state = next_states[not_done_idx]
                action = self.prev_actions[not_done_idx]
                assert action != "pass"
                action_idxs = self.vectorizer(action)
                assert action_idxs
                admissible_commands = batch_admissible_commands[not_done_idx]
                cum_reward = cum_rewards[not_done_idx]
                prev_cum_reward = self.prev_cum_rewards[not_done_idx]
                game_lost = infos["is_lost"][not_done_idx]
                done = dones[not_done_idx]
                inventory = infos["inventory"]
                desc_inventory = (
                    infos["description"][not_done_idx] + inventory[not_done_idx]
                )

                reward, exploration_bonus = self.calculate_rewards(
                    not_done_idx=not_done_idx,
                    cum_reward=cum_reward,
                    prev_cum_reward=prev_cum_reward,
                    game_lost=game_lost,
                    done=done,
                    state=desc_inventory,
                )
                self.visited_states[not_done_idx].add(desc_inventory)
                transition = Transition(
                    previous_state=previous_state,
                    next_state=next_state,
                    action=action_idxs,
                    allowed_actions=admissible_commands,
                    exploration_bonus=exploration_bonus,
                    reward=reward,
                    recipe=self.cooking_steps,
                    done=done,
                )
                # TODO: remove assertion
                assert self.prev_not_done_idxs[idx] == not_done_idx
                q_values = None
                actual_infos = {k: v[not_done_idx] for k, v in self.prev_infos.items()}
                actual_infos["action"] = self.prev_actions[not_done_idx]
                if self.prev_q_values is not None:
                    q_values = self.prev_q_values[idx]
                self.history[not_done_idx].append(
                    HistoryElement(
                        transition=transition, q_values=q_values, infos=actual_infos
                    )
                )
        else:
            assert self.current_step == 0
            for idx in range(len(infos)):
                self.visited_states[idx].add(
                    infos["description"][0] + infos["inventory"][0]
                )
        self.prev_states = next_states
        self.prev_actions = actions
        self.prev_not_done_idxs = not_done_idxs
        self.prev_cum_rewards = cum_rewards
        self.prev_q_values = self.q_values
        self.prev_infos = infos

    def calculate_rewards(
        self,
        not_done_idx,
        cum_reward: int,
        prev_cum_reward: int,
        game_lost: bool,
        done: bool,
        state: str,
    ):
        reward = float(cum_reward - prev_cum_reward)
        exploration_bonus = 0.5 * float(state not in self.visited_states[not_done_idx])
        if game_lost:
            reward = -1.5
            exploration_bonus = 0.0
        if done and cum_reward == self.max_reward:
            reward = 2.0
            exploration_bonus = 0.0
        # exploration_bonus = 0.0
        return reward, exploration_bonus

    def vectorize_state(
        self, feedback, prev_action, prev_cum_reward, infos, infos_index
    ):
        description = clean_text(infos["description"][infos_index], "description")
        feedback = clean_text(feedback, "feedback")
        inventory = parse_inventory(infos["inventory"][infos_index])
        missing_items = get_missing_ingredients_from_inventory(
            inventory, self.ingredients
        )

        admissible_commands = infos["admissible_commands"][infos_index]
        # TODO: hack?
        description = description.split("=-")[0][3:].strip()
        state_info = f" {self.vectorizer.join_symbol} ".join(
            [
                description,
                " <S> ".join(admissible_commands),
                feedback,
                " <S> ".join(inventory),
                " <S> ".join(missing_items),
                prev_action,
            ]
        )
        state_info += f" {self.vectorizer.join_symbol} {prev_cum_reward}"
        return self.vectorizer(state_info)
