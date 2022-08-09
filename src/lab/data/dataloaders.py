import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from lab.data.textclassification_datasets import *
from lab.data.datasets import *

class DataLoaderSynthetic:
    def __init__(self, device: torch.device, valid_fraction: float = .1, batch_size: int = 1, **kwargs):
        dataset = SyntheticDataset(device, **kwargs)
        n_samples = len(dataset)
        valid_len = int(n_samples * valid_fraction)
        train_len = int(n_samples - valid_len * 2)
        train_set, valid_set, test_set = random_split(dataset, [train_len, valid_len, valid_len])
        self.train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.valid = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
        self.test = DataLoader(test_set, batch_size=batch_size, shuffle=True)

        self.data_dim = self.train.dataset.dataset.data_dim
        self.n_classes = self.train.dataset.dataset.n_classes

class DataLoaderBlob:
    def __init__(self, device: torch.device, valid_fraction: float = .1, batch_size: int = 1, **kwargs):
        dataset = BlobDataset(device, **kwargs)
        n_samples = len(dataset)
        valid_len = int(n_samples * valid_fraction)
        train_len = int(n_samples - valid_len * 2)
        train_set, valid_set, test_set = random_split(dataset, [train_len, valid_len, valid_len])
        self.train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.valid = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
        self.test = DataLoader(test_set, batch_size=batch_size, shuffle=True)

        self.data_dim = self.train.dataset.dataset.data_dim
        self.n_classes = self.train.dataset.dataset.n_classes

class DataLoaderBlob3C:
    def __init__(self, device: torch.device, valid_fraction: float = .1, batch_size: int = 1, **kwargs):
        dataset = Blob3CDataset(device, **kwargs)
        n_samples = len(dataset)
        valid_len = int(n_samples * valid_fraction)
        train_len = int(n_samples - valid_len * 2)
        train_set, valid_set, test_set = random_split(dataset, [train_len, valid_len, valid_len])
        self.train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.valid = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
        self.test = DataLoader(test_set, batch_size=batch_size, shuffle=True)

        self.data_dim = self.train.dataset.dataset.data_dim
        self.n_classes = self.train.dataset.dataset.n_classes

class DataLoaderSpiral:
    def __init__(self, device: torch.device, valid_fraction: float = .1, batch_size: int = 1, **kwargs):
        seed = kwargs.get('seed', 1)
        dataset = SpiralDataset(device, **kwargs)
        n_samples = len(dataset)
        valid_len = int(n_samples * valid_fraction)
        train_len = int(n_samples - valid_len * 2)
        train_set, valid_set, test_set = random_split(dataset, [train_len, valid_len, valid_len], generator=torch.Generator().manual_seed(seed))
        self.train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.valid = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
        self.test = DataLoader(test_set, batch_size=batch_size, shuffle=True)

        self.data_dim = self.train.dataset.dataset.data_dim
        self.n_classes = self.train.dataset.dataset.n_classes

class DataLoaderCovtype:
    def __init__(self, device: torch.device, valid_fraction: float = .1, batch_size: int = 1, **kwargs):
        seed = kwargs.get('seed', 1)
        dataset = CovertypeDataset(device)
        n_samples = len(dataset)
        valid_len = int(n_samples * valid_fraction)
        train_len = int(n_samples - valid_len * 2)
        train_set, valid_set, test_set = random_split(dataset, [train_len, valid_len, valid_len], generator=torch.Generator().manual_seed(seed))
        self.train = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        self.valid = DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True)
        self.test = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

        self.data_dim = self.train.dataset.dataset.data_dim
        self.n_classes = self.train.dataset.dataset.n_classes


class DataLoaderMNIST:
    def __init__(self, device: torch.device, valid_fraction: float = .1, batch_size: int = 1, **kwargs):
        dataset = MNISTDataset(device, **kwargs)
        n_samples = len(dataset)
        valid_len = int(n_samples * valid_fraction)
        train_len = int(n_samples - valid_len * 2)
        train_set, valid_set, test_set = random_split(dataset, [train_len, valid_len, valid_len])
        self.train = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        self.valid = DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True)
        self.test = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

        self.data_dim = self.train.dataset.dataset.data_dim
        self.n_classes = self.train.dataset.dataset.n_classes

class DataLoaderYelp:
    def __init__(self, device: torch.device, valid_fraction: float = .1, batch_size: int = 1, **kwargs):
        train_set, valid_set, test_set = YelpDataset(**kwargs).get_data()
        self.train = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        self.valid = DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True)
        self.test = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

        self.data_dim = self.test.dataset.data_dim
        self.n_classes = self.test.dataset.n_classes
        self.vocab_size = self.test.dataset.vocab_size

class DataLoaderYahoo:
    def __init__(self, device: torch.device, valid_fraction: float = .1, batch_size: int = 1, **kwargs):
        train_set, valid_set, test_set = YahooDataset(**kwargs).get_data()
        self.train = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        self.valid = DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True)
        self.test = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

        self.data_dim = self.test.dataset.data_dim
        self.n_classes = self.test.dataset.n_classes
        self.vocab_size = self.test.dataset.vocab_size

class DataLoaderDB:
    def __init__(self, device: torch.device, valid_fraction: float = .1, batch_size: int = 1, **kwargs):
        train_set, valid_set, test_set = DBDataset(**kwargs).get_data()
        self.train = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        self.valid = DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True)
        self.test = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

        self.data_dim = self.test.dataset.data_dim
        self.n_classes = self.test.dataset.n_classes
        self.vocab_size = self.test.dataset.vocab_size

class DataLoaderQNLI:
    def __init__(self, device: torch.device, valid_fraction: float = .1, batch_size: int = 1, **kwargs):
        data_set = QNLIDataset(**kwargs)
        train_set, valid_set, test_set = data_set.get_data()
        self.train = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        self.valid = DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True)
        self.test = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

        self.data_dim = self.valid.dataset.data_dim
        self.n_classes = self.valid.dataset.n_classes
        self.vocab_size = self.valid.dataset.vocab_size
        self.vocab = data_set.vocab


if __name__ == '__main__':
    loader = DataLoaderQNLI(torch.device('cpu'))
    vocab = loader.vocab
    for mb in loader.test:
        sent1 = mb['input1'][0, :mb['length1']]
        sent2 = mb['input2'][0, :mb['length2']]
        print(sent1)
        print(vocab.lookup_tokens(list(sent1)))
        print(vocab.lookup_tokens(list(sent2)))
        print(mb['target'])
        print(mb['length1'])
        print(mb['length2'])



