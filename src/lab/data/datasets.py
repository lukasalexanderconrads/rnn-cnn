import torch
from torch.utils.data import Dataset
from sklearn.datasets import load_iris, fetch_covtype
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from lab.data.utils import make_spiral
import torchvision


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
        raise NotImplementedError('_get_data() is not implemented')

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

class SyntheticDatasetGaussian(ClassificationDataset):
    def __init__(self, device, n_samples=10e5):
        self.n_samples = n_samples
        super(SyntheticDatasetGaussian, self).__init__(device)

    def _get_data(self):
        class1 = torch.randn(int(self.n_samples // 2 - self.n_samples // 10), 2) - 2
        class1_cluster2 = torch.randn(int(self.n_samples // 10), 2) / 10 + 5
        class2 = torch.randn(int(self.n_samples // 2), 2) + 2

        input = torch.cat([class1, class1_cluster2, class2], dim=0)
        target = torch.cat([torch.zeros(int(self.n_samples // 2)), torch.ones(int(self.n_samples // 2))], dim=0)
        target = target.type(torch.int64)
        return input, target

class SyntheticDatasetHard(ClassificationDataset):
    def __init__(self, device, **kwargs):
        self.n_classes = kwargs.get('n_classes', 2)
        self.n_clusters_per_class = kwargs.get('n_clusters_per_class', 2)
        self.n_features = kwargs.get('n_features', 2)
        self.seed = kwargs.get('seed', 1)
        super(SyntheticDatasetHard, self).__init__(device)

    def _get_data(self):
        input, target = make_classification(n_samples=int(1e5), n_features=self.n_features,
                                            n_informative=self.n_features, n_redundant=0,
                                            n_classes=self.n_classes,
                                            n_clusters_per_class=self.n_clusters_per_class,
                                            random_state=self.seed)
        input = torch.tensor(input, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.int64)
        return input, target

class SpiralDataset(ClassificationDataset):
    def __init__(self, device, **kwargs):
        self.seed = kwargs.get('seed', 1)
        super(SpiralDataset, self).__init__(device)

    def _get_data(self):
        input, target = make_spiral(int(1e5), seed=self.seed)
        input = torch.tensor(input, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.int64)
        return input, target

class MNISTDataset(ClassificationDataset):
    def __init__(self, device, **kwargs):
        self.seed = kwargs.get('seed', 1)
        super(MNISTDataset, self).__init__(device)

    def _get_data(self):
        dataset = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                             transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                       torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
        input = dataset.data
        target = dataset.targets
        input = torch.tensor(input, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.int64)
        return input, target


if __name__ == '__main__':

    ds = SpiralDataset(torch.device('cpu'), n_samples=10e2)

    samples = ds[range(len(ds))]

    x, y = samples['input'].numpy(), samples['target'].numpy()

    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.show()




