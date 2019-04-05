from collections import defaultdict
from typing import List, Dict, Any

import torch
from textworld import EnvInfos
from agents.utils.parsing import parse_recipe
from agents.DRQN.networks.simple_net import SimpleNet
from agents.DRQN.policy.policies import GreedyPolicy
from agents.utils.params import Params
from agents.utils.parsing import get_missing_ingredients_from_inventory, parse_inventory
from agents.utils.tokenization import SpacyVectorizer
from agents.utils.types import Transition, HistoryElement
from agents.utils.utils import clean_text, idx_select


class BaseQlearningAgent:
    """ Q-learning agent that requires all available information and therefore receives maximum
    penalty
    """

    def __init__(self):
        self.config = Params.from_file("agents/configs/config.jsonnet")
        self.net = SimpleNet(
            self.config["network"], "cuda", self.config["training"]["vocab_size"]
        ).to("cuda")
        self.net.load_state_dict(self.config["training"]["model_path"])
        self.policy = GreedyPolicy()
        self.vectorizer = SpacyVectorizer()
        self.reset()

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
            extras=["recipe"],
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

        if all(dones):
            self.reset()
            return

        infos["feedback"] = observations
        infos["is_lost"] = [
            ("You lost!" in o if d else False) for o, d in zip(observations, dones)
        ]
        actions = ["pass" for _ in range(len(observations))]
        not_done_idxs = idx_select(
            list(range(len(dones))), dones, reversed_indices=True
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
        self.net.eval()
        if not_done_idxs:
            new_hidden_states, self.q_values = self.net(
                idx_select(states, not_done_idxs),
                commands_not_finished,
                recipes=[self.cooking_steps],
                hidden_states=(
                    None
                    if self.hidden_state is None
                    else torch.stack(
                        idx_select(self.hidden_state, not_done_idxs), dim=0
                    ).unsqueeze(0)
                ),
            )
            self.hidden_state = ["None" for _ in range(len(observations))]
            for idx, state in zip(not_done_idxs, new_hidden_states):
                self.hidden_state[idx] = state
            selected_action_idxs = self.policy(self.q_values)

        else:
            self.q_values = None
            selected_action_idxs = []

        for not_done_idx, adm_com, sel_act_idx in zip(
            not_done_idxs,
            idx_select(infos["admissible_commands"], not_done_idxs),
            selected_action_idxs,
        ):
            actions[not_done_idx] = adm_com[sel_act_idx]
        return actions

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
                " <S> ".join(inventory),
                " <S> ".join(missing_items),
                prev_action,
            ]
        )
        state_info += f" {self.vectorizer.join_symbol} {prev_cum_reward}"
        return (
            self.vectorizer(state_info)
            + [self.vectorizer.join_symbol_idx]
            + self.cooking_steps
        )
