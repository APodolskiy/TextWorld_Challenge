from collections import namedtuple

State = namedtuple("State", ("description", "feedback", "inventory", "prev_action"))
Transition = namedtuple(
    "Transition",
    (
        "previous_state",
        "next_state",
        "action",
        "reward",
        "exploration_bonus",
        "recipe",
        "done",
        "allowed_actions",
    ),
)
HistoryElement = namedtuple("HistoryElement", ("transition", "q_values", "infos"))
