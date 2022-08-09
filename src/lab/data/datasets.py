import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.datasets import load_iris, fetch_covtype, fetch_olivetti_faces
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from lab.data.utils import *
import torchvision



class ClassificationDataset(Dataset):
    def __init__(self, device=torch.device('cpu')):
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
    def __init__(self, device=torch.device('cpu')):
        super(CovertypeDataset, self).__init__(device)

    @staticmethod
    def _get_data():
        input, target = fetch_covtype(return_X_y=True)
        input = torch.tensor(input, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.int64)
        return input, target

class SyntheticDataset(ClassificationDataset):
    def __init__(self, device, **kwargs):
        self.n_classes = kwargs.get('n_classes', 2)
        self.n_clusters_per_class = kwargs.get('n_clusters_per_class', 2)
        self.n_features = kwargs.get('n_features', 2)
        self.n_informative = kwargs.get('n_informative', self.n_features)
        self.seed = kwargs.get('seed', 1)
        self.n_datasets = kwargs.get('n_datasets', 1)
        self.n_samples = kwargs.get('n_samples', 1e5)
        self.flip_y = kwargs.get('flip_y', 0.01)
        self.class_sep = kwargs.get('class_sep', 1)
        super(SyntheticDataset, self).__init__(device)

    def _get_data(self):

        input, target = make_classification(n_samples=int(self.n_samples), n_features=self.n_features,
                                            n_informative=self.n_informative, n_redundant=0,
                                            n_classes=self.n_classes,
                                            n_clusters_per_class=self.n_clusters_per_class,
                                            class_sep=self.class_sep,
                                            flip_y=self.flip_y,
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

class BlobDataset(ClassificationDataset):
    def __init__(self, device, **kwargs):
        self.seed = kwargs.get('seed', 1)
        self.modified = kwargs.get('modified', False)
        super(BlobDataset, self).__init__(device)

    def _get_data(self):
        input, target = make_blobs(seed=self.seed, modified=self.modified)
        input = torch.tensor(input, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.int64)
        return input, target

class Blob3CDataset(ClassificationDataset):
    def __init__(self, device, **kwargs):
        self.seed = kwargs.get('seed', 1)
        self.modified = kwargs.get('modified', False)
        super(Blob3CDataset, self).__init__(device)

    def _get_data(self):
        input, target = make_blobs3c(seed=self.seed, modified=self.modified)
        input = torch.tensor(input, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.int64)
        return input, target

class MNISTDataset(ClassificationDataset):
    def __init__(self, device, **kwargs):
        self.seed = kwargs.get('seed', 1)
        self.path = kwargs.get('path', './data')
        super(MNISTDataset, self).__init__(device)

    def _get_data(self):
        dataset = torchvision.datasets.MNIST(self.path, train=True, download=True,
                                             transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
        input = dataset.data.flatten(start_dim=1)
        target = dataset.targets
        input = torch.tensor(input, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.int64)
        return input, target



if __name__ == '__main__':

    ds = BlobDataset(torch.device('cpu'), modified=True)

    samples = ds[range(len(ds))]

    x, y = samples['input'].numpy(), samples['target'].numpy()

    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.show()




