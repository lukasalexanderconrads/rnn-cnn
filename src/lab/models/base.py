import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, **kwargs):
        super(Model, self).__init__()

        self.device = kwargs.get('device')

    def forward(self, input):
        pass


