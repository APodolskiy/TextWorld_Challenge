from pathlib import Path
from pprint import pprint
from random import shuffle
from typing import List

import gym
import textworld.gym


def parse_recipe(recipe: str):
    dir_pos = recipe.find("Directions:")
    ing_pos = recipe.find("Ingredients:")
    description = [s.strip() for s in recipe[dir_pos:].split("\n")[1:] if s]
    description.append("eat meal")
    ingredients = [s.strip() for s in recipe[ing_pos:dir_pos].split("\n")[1:] if s]
    return ingredients, description


def parse_inventory(inventory: str):
    real_inventory = inventory.split("You are carrying")[1].strip()
    if real_inventory.startswith(":"):
        return [s.strip() for s in real_inventory[2:].split("\n")]
    return []


def get_missing_ingredients_from_inventory(available_items: List, recipe_items: List):
    return [
        item
        for item in recipe_items
        if not any(obj.endswith(item) for obj in available_items)
    ]


if __name__ == "__main__":
    from agents.DRQN.custom_agent import BaseQlearningAgent

    # gamefile = "games/train_sample/tw-cooking-recipe1+cut+drop+go9-oW8msba8TGYoC6Pl.ulx"
    game_dir = Path("games/train")
    game_files = [str(d) for d in game_dir.iterdir() if d.name.endswith("ulx")]
    shuffle(game_files)
    game_files = game_files[:40]
    env_id = textworld.gym.register_games(
        game_files,
        BaseQlearningAgent.select_additional_infos(),
        max_episode_steps=100,
        name="training_par",
    )
    env = gym.make(env_id)
    for game in game_files:
        obs, infos = env.reset()
        recipe = infos["extra.recipe"]
        inventory = infos["inventory"]
        print(env.env.current_gamefile)
        print(infos["extra.walkthrough"])
        pprint(parse_recipe(recipe)[0])
        parsed_inventory = parse_inventory(inventory)
        print(parsed_inventory)
        print("MISSING:", get_missing_ingredients_from_inventory(inventory, recipe))
        print("#" * 60)
