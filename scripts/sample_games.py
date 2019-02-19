"""Folder with training games is too big, for debugging purposes it's useful to sample just some of them"""
import shutil
from pathlib import Path
from pprint import pprint

import fire
import numpy


def sample_games(files_dir, num_games):
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


def main(files_dir="games/train", num_games=10, saving_dir="games/train_sample/"):
    saving_dir = Path(saving_dir)
    saving_dir.mkdir(parents=True, exist_ok=True)
    res = sample_games(files_dir, num_games)
    for gamefile in res:
        shutil.copy(str(gamefile), str(saving_dir))
    pprint(res)


if __name__ == '__main__':
    fire.Fire(main)
