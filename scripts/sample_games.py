"""Folder with training games is too big, for debugging purposes it's useful to sample just some of them"""
from collections import defaultdict
import logging
from pathlib import Path
from pprint import pprint
import re
import shutil
from typing import Dict, List, Optional

import fire
import numpy


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


FIND_NUMBER = re.compile(r'[0-9]+$')


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


def sample_games_by_level(files_dir: str,
                          level_max: int,
                          level_min: int = 0,
                          max_games: Optional[int] = None) -> List[Path]:
    assert level_max > 0
    assert level_min >= 0
    assert (max_games is None) or (max_games > 0)

    if level_max < level_min:
        raise ValueError("Max level should be greater or equal to min level!")

    files = defaultdict(list)
    games_cnt = 0
    for file in Path(files_dir).iterdir():
        if level_min <= count_skills(file.stem) <= level_max:
            files[file.stem].append(file)
            games_cnt += 1
    file_groups = list(files.values())
    max_games = max_games or games_cnt
    return [file for file_group in file_groups[:max_games] for file in file_group]


def sample_games_by_skill(files_dir: str,
                          skill: str,
                          skill_cnt_max: int,
                          skill_cnt_min: int = 0,
                          max_games: Optional[int] = None) -> List[Path]:
    assert skill_cnt_max > 0
    assert skill_cnt_min >= 0
    assert (max_games is None) or (max_games > 0)

    files = defaultdict(list)
    games_cnt = 0
    for file in Path(files_dir).iterdir():
        skills = acquire_skills(file.stem)
        if skill in skills and skill_cnt_min <= skills[skill] <= skill_cnt_max:
            files[file.stem].append(file)
            games_cnt += 1
    file_groups = list(files.values())
    max_games = max_games or games_cnt
    return [file for file_group in file_groups[:max_games] for file in file_group]


def count_skills(game_name):
    skills = acquire_skills(game_name)
    return sum(skills.values())


def acquire_skills(game_name: str) -> Dict[str, int]:
    skill_field = game_name.split("-")[2]
    skills = skill_field.split("+")
    # TODO: all skills are differnt in the naming of files?
    return {truncate_skill(skill): get_skill_count(skill) for skill in skills}


def get_skill_count(skill: str) -> int:
    num = FIND_NUMBER.findall(skill)
    if len(num) == 0:
        return 1
    return int(num[0])


def truncate_skill(skill: str) -> str:
    num = FIND_NUMBER.findall(skill)
    if len(num) == 0:
        return skill
    return skill[:-len(num[0])]


def sample_random(files_dir="games/train", num_games=10, saving_dir="games/train_sample/"):
    saving_dir = Path(saving_dir)
    saving_dir.mkdir(parents=True, exist_ok=True)
    res = sample_random_games(files_dir, num_games)
    for gamefile in res:
        shutil.copy(str(gamefile), str(saving_dir))
    pprint(res)


def sample_by_level(files_dir="games/train"):
    pass


def sample_by_skill():
    pass


if __name__ == '__main__':
    fire.Fire()
