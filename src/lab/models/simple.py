import torch
from torch import nn

from lab.models.base import Model
from lab.utils import create_mlp
from lab.blocks import MyRNN, MyRNN2, MyRNN3, ElmanRNN

class MLP(Model):
    def __init__(self, in_dim, out_dim: int, **kwargs):
        """
        Standard MLP class
        kwargs:
            device: torch.device
            layer_dims: list of number of neurons in each layer
        """
        super().__init__(**kwargs)

        layer_dims = kwargs.get('layer_dims', [])
        layer_dims = [in_dim] + layer_dims + [out_dim]

        self.get_logits = create_mlp(layer_dims).to(self.device)

    def forward(self, input: torch.Tensor):
        """
        :param input: tensor [batch_size, input_dim]
        :return: output: tensor [batch_size, output_dim]
        """
        return self.get_logits(input)

    def train_step(self, minibatch, optimizer: torch.optim.Optimizer):
        input = minibatch['input']
        target = minibatch['target']

        optimizer.zero_grad()

        logits = self.forward(input)

        loss_stats = self.loss(logits, target)

        loss_stats['loss'].backward()
        optimizer.step()

        return loss_stats

    def test_step(self, minibatch):
        input = minibatch['input']
        target = minibatch['target']

        logits = self.forward(input)

        loss_stats = self.loss(logits, target)
        metrics = self.metrics(logits, target)

        return metrics | loss_stats

    def evaluate(self, minibatch, **kwargs):
        return self.test_step(minibatch)

    @staticmethod
    def loss(logits, target):
        loss = nn.functional.cross_entropy(logits, target)
        return {'loss': loss, 'cross_entropy': loss}

    def metrics(self, logits, target):
        accuracy = self.get_accuracy(logits, target)
        return {'accuracy': accuracy}

    @staticmethod
    def get_accuracy(logits, target):
        prediction = torch.argmax(logits, dim=1)
        accuracy = torch.mean((prediction == target).float())
        return accuracy

    def print_layers(self):
        print('model layers:')
        print(self.get_logits)