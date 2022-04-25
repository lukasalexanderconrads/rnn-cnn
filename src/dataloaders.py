import torch
from torch.utils.data import DataLoader, Dataset, random_split


class DataLoaderRandomSplit:
    def __init__(self, dataset: Dataset, test_fraction: float = .1, batch_size: int = 1):
        data_len = len(dataset)
        train_len = int(data_len * (1 - test_fraction))
        test_len = data_len - train_len

        train_set, test_set = random_split(dataset, [train_len, test_len])

        self.train = DataLoader(train_set, batch_size=batch_size)
        self.test = DataLoader(test_set, batch_size=batch_size)





