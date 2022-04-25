import torch
from torch import nn

from src.models.base import Model


class MLP(Model):
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        """
        Standard MLP class
        kwargs:
            device: torch.device
            layer_dims: list of number of neurons in each layer
        """
        super().__init__(**kwargs)
        layer_dims = kwargs.get('layer_dims')
        layer_dims = [in_dim] + layer_dims + [out_dim]

        n_layers = len(layer_dims)
        layers = []
        for layer_idx in range(n_layers - 1):
            layers.append(nn.Linear(layer_dims[layer_idx], layer_dims[layer_idx + 1]))
            if layer_idx != n_layers - 2:
                layers.append(nn.ReLU())

        self.get_logits = nn.Sequential(*layers).to(self.device)

    def forward(self, input: torch.Tensor):
        """
        :param input: tensor [batch_size, input_dim]
        :return: output: tensor [batch_size, output_dim]
        """
        if input.dim() == 1:
            input = input.unsqueeze(0)
        return self.get_logits(input)

    def train_step(self, minibatch, optimizer: torch.optim.Optimizer):
        input = minibatch['input']
        target = minibatch['target']

        optimizer.zero_grad()

        prediction = self.forward(input)

        loss = self.get_loss(prediction, target)

        loss.backward()
        optimizer.step()

        return {'loss': loss}

    def test_step(self, minibatch):
        input = minibatch['input']
        target = minibatch['target']

        logits = self.forward(input)

        loss = self.get_loss(logits, target)
        accuracy = self.get_accuracy(logits, target)

        return {'loss': loss, 'accuracy': accuracy}

    @staticmethod
    def get_loss(logits, target):
        return nn.functional.cross_entropy(logits, target)

    @staticmethod
    def get_accuracy(logits, target):
        prediction = torch.argmax(logits, dim=1)
        accuracy = torch.mean((prediction == target).float())
        return accuracy


class RNN(MLP):
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        """
        MLP with feed forward layers and recurrent layers
        :param kwargs:
            device: torch.device
            fc_dims: list of number of neurons for each layer in feed forward part
            rnn_dim: int, number of neurons of in rnn layer
            out_dim: int, number of output neurons
        """
        super(MLP, self).__init__(**kwargs)
        fc_dims = kwargs.get('fc_dims')
        rnn_dim = kwargs.get('rnn_dim')

        fc_dims = [in_dim] + fc_dims + [rnn_dim]
        n_layers = len(fc_dims)
        layers = []
        for layer_idx in range(n_layers - 1):
            layers.append(nn.Linear(fc_dims[layer_idx], fc_dims[layer_idx + 1]))
            layers.append(nn.ReLU())

        self.fc_layers = nn.Sequential(*layers).to(self.device)

        self.rnn_layer = nn.Sequential(nn.Linear(rnn_dim, rnn_dim),
                                       nn.ReLU()).to(self.device)

        self.get_logits = nn.Linear(rnn_dim, out_dim).to(self.device)

    def forward(self, input, max_rec=1):
        """
        :param input: tensor [batch_size, input_dim]
        :return: output: tensor [batch_size, output_dim]
        """
        if input.dim() == 1:
            input = input.unsqueeze(0)

        h = self.fc_layers(input)
        for _ in range(max_rec):
            h = self.rnn_layer(h)

        return self.get_logits(h)






if __name__ == '__main__':
    mlp = MLP([10, 20, 20, 20, 5], device=torch.device('cpu'))
    print(mlp.get_logits)

    x = torch.randn((16, 10))
    y = mlp.forward(x)
    print(y.size())

    rnn = RNN([10, 20], 20, 5, device=torch.device('cpu'))
    print(rnn.get_logits)

    x = torch.randn((16, 10))
    y = rnn.forward(x)
    print(y.size())