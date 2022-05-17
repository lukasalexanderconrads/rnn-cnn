import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


class DataLoaderRandomSplit:
    def __init__(self, dataset: Dataset, valid_fraction: float = .1, batch_size: int = 1):

        data_len = len(dataset)
        valid_len = int(data_len * valid_fraction)
        train_len = data_len - valid_len*2

        train_set, valid_set, test_set = random_split(dataset, [train_len, valid_len, valid_len])

        self.train = DataLoader(train_set, batch_size=batch_size)
        self.valid = DataLoader(valid_set, batch_size=batch_size)
        self.test = DataLoader(test_set, batch_size=batch_size)

        self.data_dim = self.train.dataset.dataset.data_dim
        self.n_classes = self.train.dataset.dataset.n_classes


class DataLoaderCifar:
    def __init__(self, valid_fraction: float = .1, batch_size: int = 1):

        train_dataset = CIFAR10(root='data/', transform=ToTensor(), download=True)
        data_len = len(train_dataset)
        train_len = int(data_len * (1 - valid_fraction))
        test_len = data_len - train_len
        train_set, test_set = random_split(train_dataset, [train_len, test_len])


        self.train = DataLoader(train_set, batch_size=batch_size)
        self.valid = DataLoader(test_set, batch_size=batch_size)

        self.test = DataLoader(CIFAR10(root='data/', train=False, transform=ToTensor()), batch_size=batch_size)

        self.data_dim = self.test.dataset.data[0].shape
        self.n_classes = len(self.test.dataset.classes)



if __name__ == '__main__':
    print(DataLoaderCifar())

