import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris, fetch_covtype


class ClassificationDataset(Dataset):
    def __init__(self, device):
        super(ClassificationDataset, self).__init__()

        input, target = self._get_data()

        self.input = input.to(device)
        self.target = target.to(device)

        self.data_dim = self.input.size(1)
        self.n_classes = int(max(self.target) + 1)

    def _get_data(self):
        pass

    def __getitem__(self, item):
        return {'input': self.input[item],
                'target': self.target[item]}

    def __len__(self):
        return self.input.size(0)


class IrisDataset(ClassificationDataset):
    def __init__(self, device):
        super(IrisDataset, self).__init__(device)

    @staticmethod
    def _get_data():
        input, target = load_iris(return_X_y=True)
        input = torch.tensor(input, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.int64)
        return input, target


class CovertypeDataset(ClassificationDataset):
    def __init__(self, device):
        super(CovertypeDataset, self).__init__(device)

    @staticmethod
    def _get_data():
        input, target = fetch_covtype(return_X_y=True)
        input = torch.tensor(input, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.int64)
        return input, target










if __name__ == '__main__':

    dataset = IrisDataset(torch.device('cpu'))

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for minibatch in loader:
        print(minibatch['input'].size())


