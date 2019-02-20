"""Folder with training games is too big, for debugging purposes it's useful to sample just some of them"""
from pathlib import Path
from pprint import pprint
import re
import shutil
from typing import List, Optional

import fire
import numpy


FIND_NUMBER = re.compile(r'[0-9]')


def sample_random_games(files_dir: str, num_games: int) -> List[Path]:
    numpy.random.seed(0)
    files = sorted(Path(files_dir).iterdir())
    # each game is defined by .json, .z8, .ulx files
    total_games = len(files) // 3
    sampled_games = numpy.random.choice(total_games, num_games, replace=False)
    indices = numpy.sort(sampled_games)
    sampled_files = []
    for idx in indices:
        sampled_files.extend(files[idx * 3: idx * 3 + 3])
    return sampled_files


def sample_games_by_level(files_dir: str, level: int, max_games: Optional[int] = None) -> List[Path]:
    files = sorted(Path(files_dir).iterdir())
    ulx_files = [file for file in files if file.suffix == '.ulx']
    pass


def acquire_skills(game_name: str):
    skill_field = game_name.split("-")[2]
    skills = skill_field.split("+")
    return [(skill, get_skill_count(skill)) for skill in skills]


def get_skill_count(skill: str) -> int:
    num = FIND_NUMBER.findall(skill)
    if len(num) == 0:
        return 1
    return int(num[0])


def main(files_dir="games/train", num_games=10, saving_dir="games/train_sample/"):
    saving_dir = Path(saving_dir)
    saving_dir.mkdir(parents=True, exist_ok=True)
    res = sample_random_games(files_dir, num_games)
    for gamefile in res:
        shutil.copy(str(gamefile), str(saving_dir))
    pprint(res)


if __name__ == '__main__':
    fire.Fire(main)
