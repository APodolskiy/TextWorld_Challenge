import fire


def split_train_dev(train_data_path: str, dev_folder: str) -> None:
    """
    Splits whole dataset into train and dev datasets.
    !!!Important!!! For the space and time performance
    this function actually moves some files from the train
    folder into dev folder. File sampling is random
    :param train_data_path: path to the folder with training data
    :param dev_folder: path to the folder where dev data would be stored.
    """
    pass


if __name__ == "__main__":
    fire.Fire(split_train_dev)
