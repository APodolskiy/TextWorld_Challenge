import fire
from pathlib import Path
from pprint import pprint
import shutil
from typing import Union

from scripts.sample_games import sample_random_games


def split_train_dev(train_data_path: str, dev_data_path: str, valid_frac: Union[int, float] = 700) -> None:
    """
    Splits whole dataset into train and dev datasets.
    !!!Important!!! For the space and time performance
    this function actually moves some files from the train
    folder into dev folder. File sampling is random
    :param train_data_path: path to the folder with training data
    :param dev_data_path: path to the folder where dev data would be stored.
    """
    print(f"Do you really want to create validation data from {train_data_path} and store it in {dev_data_path}\n"
          f"Be aware that game files will be MOVED from the source folder to the target folder!\n"
          f"If it's intended type in yes")
    x = input()
    if x != 'yes':
        print("Aborting creation of validation data!")
        exit(1)
    train_dir_path = Path(train_data_path)
    games_num = sum([1 for f in train_dir_path.iterdir() if f.suffix == '.ulx'])
    if isinstance(valid_frac, float):
        if not 0 < valid_frac < 1:
            raise ValueError(f"Inappropriate validation dataset fraction: {valid_frac}.\n"
                             f"Specify number from the interval (0, 1).")
        valid_games_num = games_num * valid_frac
    elif isinstance(valid_frac, int):
        valid_games_num = valid_frac
    else:
        raise ValueError(f"Incorrect type of valid_frac argument. Should be int or float.")
    game_files = sample_random_games(train_data_path, num_games=valid_games_num)
    dev_data_path = Path(dev_data_path)
    dev_data_path.mkdir(parents=True, exist_ok=False)
    for game_file in game_files:
        shutil.move(str(game_file), str(dev_data_path))
    pprint(game_files)


if __name__ == "__main__":
    fire.Fire(split_train_dev)
