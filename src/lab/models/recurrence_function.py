import torch
from torch import nn

from lab.models.simple import RNN, LSTM
from lab.blocks import *
from lab.utils import create_mlp, create_rbf

class LearnableRNN(RNN):

    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        """
        RNN with learnable function for number of recurrences
        :param kwargs:
        """
        super(LearnableRNN, self).__init__(in_dim, out_dim, **kwargs)

        # recurrence function
        rec_fn_layers = kwargs.get('rec_fn_layers', [])
        self.rec_fn_input = kwargs.get('rec_fn_input', 'input') # input, embedding
        rec_fn_in_dim = in_dim if self.rec_fn_input == 'input' else self.rnn_dim

        rec_fn_layers = [rec_fn_in_dim] + rec_fn_layers + [self.max_rec]
        self.get_recurrences = create_mlp(rec_fn_layers).to(self.device)

        self.tau = kwargs.get('tau', .75)

        #self.get_recurrences = create_rbf(rec_fn_layers).to(self.device)

        # rnn type
        self.rnn_type = kwargs.get('rnn_type', 'linear')    # linear, lstm, gru, myrnn1, myrnn2, myrnn3, elman
        self._get_rnn_layer()

        # regularization
        self.reg_target = torch.tensor(kwargs.get('reg_target', [.5, .5])).unsqueeze(0).to(self.device)
        self.reg_weight = kwargs.get('reg_weight', 1)

    def forward(self, input):
        """
        :param input: tensor [batch_size, input_dim]
        :return: output: tensor [batch_size, output_dim]
        """
        if input.dim() == 1:
            input = input.unsqueeze(0)

        a = self.fc_layers(input)
        if self.rec_fn_input == 'embedding':
            a_rec_fn = a.clone()
        elif self.rec_fn_input == 'embedding_detached':
            a_rec_fn = a.clone().detach()
        elif self.rec_fn_input == 'input':
            a_rec_fn = input

        if self.rnn_type == 'linear':
            l = a.clone()
        else:
            h, c = None, None
        logits_list = []
        for step in range(self.max_rec):
            # recurrent layer
            if self.rnn_type == 'lstm':
                l, (h, c) = self.rnn_layer(a.unsqueeze(0), (h, c))
                l = l.squeeze()
            elif self.rnn_type == 'gru':
                l, h = self.rnn_layer(a.unsqueeze(0), h)
                l = l.squeeze()
            elif self.rnn_type == 'linear':
                l = self.rnn_layer(l)
            else:
                l, h = self.rnn_layer(a, h)
            # get class probs
            logits = self.get_logits(l)
            logits_list.append(logits)

        final_step_logits = self.get_recurrences(a_rec_fn)    # [batch_size, max_rec]

        final_step_probs = nn.functional.gumbel_softmax(final_step_logits, dim=-1, tau=self.tau, hard=True)

        logits_stacked = torch.stack(logits_list)           # [max_rec, batch_size, n_classes]

        return logits_stacked, final_step_probs, final_step_logits

    def train_step(self, minibatch, optimizer: torch.optim.Optimizer):
        input = minibatch['input']
        target = minibatch['target']

        optimizer.zero_grad()

        logits_stacked, final_step_probs, final_step_logits = self.forward(input)

        loss_stats = self.loss(logits_stacked, target, final_step_probs)

        loss_stats['loss'].backward()

        optimizer.step()

        return loss_stats

    def test_step(self, minibatch):
        input = minibatch['input']
        target = minibatch['target']

        logits_stacked, final_step_probs, final_step_logits = self.forward(input)

        loss_stats = self.loss(logits_stacked, target, final_step_probs)
        metrics = self.metrics(logits_stacked, target, final_step_probs)

        return loss_stats | metrics

    def loss(self, logits_stacked, target, final_step_probs):
        """
        computes loss
        :param logits_stacked: logits for each example at each step [max_rec, batch_size, n_classes]
        :param target: target classes [batch_size]
        :param final_step_probs: indices corresponding to the final step for each example [batch_size, max_rec]
        :return:
        """

        probs_stacked = nn.functional.softmax(logits_stacked, dim=-1)
        weighted_probs = torch.sum(probs_stacked * final_step_probs.T.unsqueeze(-1), dim=0)
        mask = (weighted_probs <= 1e-10).float()
        logits = torch.log(weighted_probs + mask * 1e-10)

        cross_entropy = nn.functional.nll_loss(logits, target)

        if self.reg_weight != 0:
            reg_loss = self._get_reg_loss(final_step_probs)
        else:
            reg_loss = 0

        loss = cross_entropy + self.reg_weight * reg_loss
        return {'loss': loss, 'cross_entropy': cross_entropy, 'reg_loss': reg_loss}

    def metrics(self, logits_stacked, target, final_step_probs):
        final_steps = torch.argmax(final_step_probs, dim=-1)

        logits = self.get_final_logits(logits_stacked, final_steps)
        accuracy = self.get_accuracy(logits, target)
        # compute average number of recurrences required
        avg_steps = torch.mean(final_steps.float()) + 1

        loss_type = ['first', 'every', 'final'].index(self.loss_type)

        return {'accuracy': accuracy,
                'average steps': avg_steps,
                'loss_type': loss_type}

    def _get_rnn_layer(self): # linear, lstm, gru, myrnn1, myrnn2, myrnn3, elman
        if self.rnn_type == 'lstm':
            self.rnn_layer = nn.LSTM(input_size=self.rnn_dim,
                                  hidden_size=self.rnn_dim).to(self.device)
        elif self.rnn_type == 'gru':
            self.rnn_layer = nn.GRU(input_size=self.rnn_dim,
                                     hidden_size=self.rnn_dim).to(self.device)
        elif self.rnn_type == 'myrnn1':
            self.rnn_layer = MyRNN(self.rnn_dim, self.rnn_dim, self.device)
        elif self.rnn_type == 'myrnn2':
            self.rnn_layer = MyRNN2(self.rnn_dim, self.device)
        elif self.rnn_type == 'myrnn3':
            self.rnn_layer = MyRNN3(self.rnn_dim, self.device)
        elif self.rnn_type == 'elman':
            self.rnn_layer = ElmanRNN(self.rnn_dim, self.rnn_dim, self.device)
        else:
            pass

    def _get_reg_loss(self, final_step_probs):
        """
        cross entropy between reg target and average final step probs
        """
        avg_final_step_probs = torch.mean(final_step_probs, dim=0)
        #mask = (avg_final_step_probs <= 0.001).float()
        #log_probs = torch.log(avg_final_step_probs + mask * 0.001)
        #reg_loss = -torch.sum(log_probs * self.reg_target)
        reg_loss = torch.sum((avg_final_step_probs - self.reg_target)**2)

        return reg_loss
