import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris, fetch_covtype
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt


class ClassificationDataset(Dataset):
    def __init__(self, device):
        super(ClassificationDataset, self).__init__()

        input, target = self._get_data()

        self.input = input.to(device)
        self.target = target.to(device)

        self.data_dim = self.input.size(1)
        self.n_classes = int(max(self.target) + 1)

    @staticmethod
    def _get_data():
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

class SyntheticDataset(ClassificationDataset):
    def __init__(self, device, n_samples=10e5):
        self.n_samples = n_samples
        super(SyntheticDataset, self).__init__(device)

    def _get_data(self):
        #input, target = make_classification(n_samples=int(self.n_samples), n_features=2,
        #                                    n_informative=2, n_redundant=0,
        #                                    random_state=1)

        class1 = torch.randn(int(self.n_samples // 2 - self.n_samples // 10), 2) - 2
        class1_cluster2 = torch.randn(int(self.n_samples // 10), 2) / 10 + 5
        class2 = torch.randn(int(self.n_samples // 2), 2) + 2

        input = torch.cat([class1, class1_cluster2, class2], dim=0)
        target = torch.cat([torch.zeros(int(self.n_samples // 2)), torch.ones(int(self.n_samples // 2))], dim=0)
        target = target.type(torch.int64)
        return input, target


if __name__ == '__main__':

    ds = SyntheticDataset(torch.device('cpu'), n_samples=10e2)

    samples = ds[range(len(ds))]

    x, y = samples['input'].numpy(), samples['target'].numpy()

    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.show()




