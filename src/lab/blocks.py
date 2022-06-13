import torch
from torch import nn

class MyRNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, device):
        super(MyRNN, self).__init__()
        self.device = device
        self.out_gate = nn.Sequential(nn.Linear(in_dim + hidden_dim, in_dim),
                                      nn.LeakyReLU()).to(self.device)
        self.hidden_gate = nn.Sequential(nn.Linear(in_dim + hidden_dim, hidden_dim),
                                         nn.LeakyReLU()).to(self.device)

        self.hidden_dim = hidden_dim

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros((x.size(0), self.hidden_dim), device=self.device)

        gate_input = torch.cat((x, h), dim=1)

        y = self.out_gate(gate_input)
        h = self.hidden_gate(gate_input)

        return y, h


class MyRNN2(nn.Module):
    def __init__(self, dim, device):
        super(MyRNN2, self).__init__()
        self.device = device
        self.out_gate = nn.Sequential(nn.Linear(dim, dim),
                                      nn.LeakyReLU()).to(self.device)
        self.hidden_gate = nn.Sequential(nn.Linear(dim, dim),
                                         nn.Sigmoid()).to(self.device)

        self.dim = dim

    def forward(self, x, h=None):
        if h is None:
            h = torch.ones((x.size(0), self.dim), device=self.device)

        gate_input = x * h

        y = self.out_gate(gate_input)
        h = self.hidden_gate(gate_input)

        return y, h