"""Folder with training games is too big, for debugging purposes it's useful to sample just some of them"""
from collections import defaultdict
import logging
from pathlib import Path
from pprint import pprint
import re
import shutil
from typing import Dict, List, Optional, Callable, Any

import fire
import numpy


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


FIND_NUMBER = re.compile(r'[0-9]+$')


def sample_random_games(files_dir: str, num_games: int) -> List[Path]:
    """
    Sample random games from the specified folder
    Args:
        files_dir: path to the directory with games
        num_games: number of games to sample

    Returns: list of `pathlib.Path` related to the sampled games

    """
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
    """
    Sample games by level
    :param files_dir: path to the directory with game files
    :param level_max: maximum game level
    :param level_min: minimum game level
    :param max_games: maximum number of games to sample
    :return: list of `pathlib.Path` objects related to the game files
    """
    assert level_max > 0
    assert level_min >= 0
    assert (max_games is None) or (max_games > 0)

    if level_max < level_min:
        raise ValueError("Max level should be greater or equal to min level!")

    class LevelCmp:
        def __init__(self, level_max: int, level_min: int):
            self.level_max = level_max
            self.level_min = level_min

        def __call__(self, game_name: str) -> bool:
            level = count_skills(game_name)
            if self.level_min <= level <= self.level_max:
                return True
            return False

    return filter_games(files_dir=files_dir,
                        func_cmp=LevelCmp(level_max, level_min),
                        max_games=max_games)


def sample_games_by_skill(files_dir: str,
                          skill: str,
                          skill_cnt_max: int,
                          skill_cnt_min: int = 0,
                          max_games: Optional[int] = None) -> List[Path]:
    """
    Sample games by skill level from spcified directory.
    :param files_dir: path to the directory with games
    :param skill: skill name
    :param skill_cnt_max: maximum skill level
    :param skill_cnt_min: minimum skill level
    :param max_games: maximum number of games to sample
    :return: list of `pathlib.Path` objects related to the game files.
    """
    assert skill_cnt_max > 0
    assert skill_cnt_min >= 0
    assert (max_games is None) or (max_games > 0)

    if skill_cnt_max < skill_cnt_min:
        raise ValueError("Max skill level should be greater or equal to min skill level!")

    class SkillLevelCmp:
        def __init__(self, skill: str, skill_cnt_max: int, skill_cnt_min: int):
            self.skill = skill
            self.skill_cnt_max = skill_cnt_max
            self.skill_cnt_min = skill_cnt_min

        def __call__(self, game_name: str) -> bool:
            skill_level = count_skill(game_name, self.skill)
            if self.skill_cnt_min <= skill_level <= self.skill_cnt_max:
                return True
            return False

    return filter_games(files_dir=files_dir,
                        func_cmp=SkillLevelCmp(skill, skill_cnt_max, skill_cnt_min),
                        max_games=max_games)


def filter_games(files_dir: str, func_cmp: Callable[[str], bool], max_games: int):
    numpy.random.seed(0)
    files = defaultdict(list)
    games_cnt = 0
    for game_file in Path(files_dir).iterdir():
        if func_cmp(game_file.stem):
            files[game_file.stem].append(game_file)
            games_cnt += 1
    file_groups = numpy.array(list(files.values()))
    numpy.random.shuffle(file_groups)
    max_games = max_games or games_cnt
    return [game_file for file_group in file_groups[:max_games] for game_file in file_group]


def count_skills(game_name: str) -> int:
    """
    Count skills needed for the game.
    :param game_name: game name
    :return: number of skill in the game
    """
    skills = acquire_skills(game_name)
    return sum(skills.values())


def acquire_skills(game_name: str) -> Dict[str, int]:
    """
    Acquire needed skills from the games name.
    :param game_name: game name
    :return: dict with skill as a key and skill count as a value
    """
    skill_field = game_name.split("-")[2]
    skills = skill_field.split("+")
    return {truncate_skill(skill): get_skill_count(skill) for skill in skills}


def count_skill(game_name: str, skill: str) -> int:
    """
    Count skill level of the game.
    :param game_name: game name
    :param skill: skill name
    :return: skill level
    """
    skills = acquire_skills(game_name)
    if skill in skills:
        return skills[skill]
    return 0


def get_skill_count(skill: str) -> int:
    """
    Count the skill
    :param skill: raw skill name
    :return: skill's count
    """
    num = FIND_NUMBER.findall(skill)
    if len(num) == 0:
        return 1
    return int(num[0])


def truncate_skill(skill: str) -> str:
    """
    Truncate skill's name.

    :param skill: raw skill name
    :return: truncated skill name

    -------
    Example:
    'goal6' will be transformed into 'goal'
    'goal12' will be transformed into 'goal'
    'goal' will be transformed into 'goal'
    """
    num = FIND_NUMBER.findall(skill)
    if len(num) == 0:
        return skill
    return skill[:-len(num[0])]


class SampleGames:
    """
    Script's entry point.
    """
    def __init__(self,
                 files_dir: str = "games/train",
                 saving_dir: str = "games/train_sample/",
                 force: bool = False):
        self.files_dir = files_dir
        self.saving_dir = saving_dir
        self.force = force
        self._create_saving_dir()

    def _create_saving_dir(self) -> None:
        """
        Create directory where game files will be stored.
        """
        saving_dir = Path(self.saving_dir)
        if saving_dir.exists() and self.force:
            shutil.rmtree(str(saving_dir))
        saving_dir.mkdir(parents=True, exist_ok=False)

    def _copy_games(self, games: List[Path]) -> None:
        """
        Copy games into directory
        :param games: list of `pathlib.Path` related to the game files
        """
        for gamefile in games:
            shutil.copy(str(gamefile), str(self.saving_dir))
        pprint(games)

    def random(self, num_games: int = 10) -> None:
        """
        Sample random games
        :param num_games: number of games to sample
        :return:
        """
        games = sample_random_games(self.files_dir, num_games)
        self._copy_games(games)

    def level(self, level_max: int, level_min: int = 0, max_games: Optional[int] = None) -> None:
        """
        Sample games by level
        :param level_max: maximum level
        :param level_min: minimum level
        :param max_games: maximum number of games to sample
        """
        games = sample_games_by_level(files_dir=self.files_dir,
                                      level_max=level_max,
                                      level_min=level_min,
                                      max_games=max_games)
        self._copy_games(games)

    def skill(self, skill: str, skill_cnt_max: int, skill_cnt_min: int = 0, max_games: Optional[int] = None) -> None:
        """
        Sample games by specific skill
        :param skill: skill name
        :param skill_cnt_max: maximum level of skill
        :param skill_cnt_min: minimum level of skill
        :param max_games: maximum number of games to sample
        """
        games = sample_games_by_skill(files_dir=self.files_dir,
                                      skill=skill,
                                      skill_cnt_max=skill_cnt_max,
                                      skill_cnt_min=skill_cnt_min,
                                      max_games=max_games)
        self._copy_games(games)


if __name__ == '__main__':
    fire.Fire(SampleGames)
