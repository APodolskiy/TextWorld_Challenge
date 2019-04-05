from itertools import zip_longest
from typing import List

from agents.utils.types import HistoryElement


def get_sample_history_trace(history: List[HistoryElement], gamefile):
    traces = history[0]
    result = [gamefile]
    prev_reward = 0
    prev_exp_reward = 0
    cum_reward = 0
    for step_num, game_step in enumerate(traces):
        result.append(
            f"{game_step.infos['feedback'].strip()}\n"
            f"Finished step {step_num}. Reward={prev_reward} Exploration reward={prev_exp_reward}"
        )
        qvalues = game_step.q_values
        info = game_step.infos
        transition = game_step.transition
        chosen_action = info["action"]
        if qvalues is None:
            result.append(f" > Random choice: {chosen_action.upper()}")
        else:
            admissible_commands = info["admissible_commands"]
            for command, q_value in zip_longest(
                admissible_commands, qvalues, fillvalue=None
            ):
                if command == chosen_action:
                    command = command.upper()
                result.append(f" > {command}: {q_value:.3f}")
        prev_reward = transition.reward
        prev_exp_reward = transition.exploration_bonus
        cum_reward += prev_reward == 1.0
    result.append(f"Reward={prev_reward} Exploration reward={prev_exp_reward}")
    result.append(f"Finished in {len(traces)} steps. Total reward = {cum_reward}")
    return "\n".join(result)
