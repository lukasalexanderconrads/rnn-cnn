import torch
from torch.utils.data import DataLoader, Subset
from lab.data.datasets import SyntheticDatasetHard, SpiralDataset, CovertypeDataset
from sklearn.model_selection import StratifiedKFold

class DataLoaderCV:
    def __init__(self, device: torch.device, batch_size: int = 1, **kwargs):
        self.batch_size = batch_size
        self.n_splits = kwargs.get('n_splits', 10)
        seed = kwargs.get('seed', 1)

        k_fold = StratifiedKFold(n_splits=self.n_splits, random_state=seed, shuffle=True)
        self.dataset = self._get_dataset(device, **kwargs)
        self.split_iter = k_fold.split(self.dataset.input.cpu(), self.dataset.target.cpu())

        self.data_dim = self.dataset.data_dim
        self.n_classes = self.dataset.n_classes

    def make_split(self):
        train_index, test_index = next(self.split_iter)
        train_set = Subset(self.dataset, train_index)
        test_set = Subset(self.dataset, test_index)

        self.train = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        self.test = DataLoader(test_set, batch_size=self.batch_size, shuffle=True)

    def _get_dataset(self, device, **kwargs):
        raise NotImplementedError('_get_dataset() is not implemented')


class DataLoaderSyntheticHardCV(DataLoaderCV):
    def __init__(self, device: torch.device, batch_size: int = 1, **kwargs):
        super(DataLoaderSyntheticHardCV, self).__init__(device, batch_size, **kwargs)

    def _get_dataset(self, device, **kwargs):
        return SyntheticDatasetHard(device, **kwargs)

class DataLoaderSpiralCV(DataLoaderCV):
    def __init__(self, device: torch.device, batch_size: int = 1, **kwargs):
        super(DataLoaderSpiralCV, self).__init__(device, batch_size, **kwargs)

    def _get_dataset(self, device, **kwargs):
        return SpiralDataset(device, **kwargs)

class DataLoaderCovertypeCV(DataLoaderCV):
    def __init__(self, device: torch.device, batch_size: int = 1, **kwargs):
        super(DataLoaderCovertypeCV, self).__init__(device, batch_size, **kwargs)

    def _get_dataset(self, device, **kwargs):
        return CovertypeDataset(device)