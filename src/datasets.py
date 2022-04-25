import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris

class IrisDataset(Dataset):
    def __init__(self, device):
        super(IrisDataset).__init__()

        input, target = load_iris(return_X_y=True)
        self.input = torch.tensor(input, device=device, dtype=torch.float32)
        self.target = torch.tensor(target, device=device, dtype=torch.int64)

        self.data_dim = self.input.size(1)
        self.n_classes = int(max(self.target) + 1)

    def __getitem__(self, item):
        return {'input': self.input[item],
                'target': self.target[item]}

    def __len__(self):
        return self.input.size(0)








if __name__ == '__main__':

    dataset = IrisDataset(torch.device('cpu'))

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for minibatch in loader:
        print(minibatch['input'].size())


