from textworld import EnvInfos

REQUIRED_INFOS = EnvInfos(
    max_score=True,
    description=True,
    inventory=True,
    extras=["walkthrough", "recipe"],
    admissible_commands=True,
)
