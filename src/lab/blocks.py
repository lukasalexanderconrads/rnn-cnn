import torch
from torch import nn
import math

class MyRNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, device):
        super(MyRNN, self).__init__()
        self.device = device
        self.out_gate = nn.Sequential(nn.Linear(in_dim + hidden_dim, in_dim),
                                      nn.LeakyReLU()).to(self.device)
        self.hidden_gate = nn.Sequential(nn.Linear(in_dim + hidden_dim, hidden_dim),
                                         nn.LeakyReLU()).to(self.device)
        self.in_dim = in_dim
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

class MyRNN3(nn.Module):
    def __init__(self, dim, device):
        super(MyRNN3, self).__init__()
        self.device = device
        self.out_gate = nn.Sequential(nn.Linear(dim, dim),
                                      nn.LeakyReLU()).to(self.device)
        self.hidden_gate = nn.Sequential(nn.Linear(dim, dim),
                                         nn.LeakyReLU()).to(self.device)

        self.dim = dim

    def forward(self, x, h=None):
        if h is None:
            h = torch.ones((x.size(0), self.dim), device=self.device)

        gate_input = x + h

        y = self.out_gate(gate_input)
        h = self.hidden_gate(gate_input)

        return y, h

class ElmanRNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, device, simplified=False):
        super(ElmanRNN, self).__init__()
        self.device = device
        self.hidden_gate = nn.Sequential(nn.Linear(in_dim + hidden_dim, hidden_dim),
                                      nn.LeakyReLU()).to(self.device)
        if not simplified:
            self.out_gate = nn.Sequential(nn.Linear(hidden_dim, in_dim),
                                             nn.LeakyReLU()).to(self.device)

        self.hidden_dim = hidden_dim
        self.simplified = simplified

    def forward(self, x, h=None):
        if h is None:
            h = torch.ones((x.size(0), self.hidden_dim), device=self.device)

        gate_input = torch.cat((x, h), dim=1)

        h = self.hidden_gate(gate_input)
        if self.simplified:
            return h, h
        y = self.out_gate(h)

        return y, h


class EncoderLSTM(nn.Module):
    def __init__(self, device, vocab_size, **kwargs):
        super(EncoderLSTM, self).__init__()
        self.emb_dim = kwargs.get('emb_dim', 32)
        self.embedding = nn.Embedding(vocab_size+1, self.emb_dim, padding_idx=vocab_size).to(device)

        self.hidden_dim = kwargs.get('hidden_dim', self.emb_dim)
        self.dropout = kwargs.get('dropout', 0)
        num_layers = kwargs.get('num_layers', 1)
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_dim, dropout=self.dropout, num_layers=num_layers).to(device)
        #self.lstm = nn.RNN(self.emb_dim, self.hidden_dim, dropout=self.dropout, num_layers=num_layers).to(device)


    def forward(self, x, seq_len):
        """
        :param x: batch of sequences shape [batch_size, fix_len]
        :return: final hidden state of lstm
        """
        x_emb = self.embedding(x)

        x_packed = nn.utils.rnn.pack_padded_sequence(x_emb, seq_len, batch_first=True, enforce_sorted=False)

        _, (hidden_states, _) = self.lstm(x_packed)
        #_, hidden_states = self.lstm(x_packed)

        last_hidden_state = hidden_states[-1]
        return last_hidden_state

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x).transpose(0, 1)