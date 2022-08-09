import torch
from torch import nn
from lab.models.simple import MLP
from lab.blocks import *
from lab.utils import create_mlp, create_rbf
from lab.utils import create_instance

class RNN(MLP):

    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        """
        RNN with learnable function for number of recurrences
        :param kwargs:
        """
        super(MLP, self).__init__(**kwargs)

        ### NETWORK STRUCTURE ###
        self.rnn_dim = kwargs.get('rnn_dim')
        self.hidden_dim = kwargs.get('hidden_dim', self.rnn_dim)

        # feature embedding
        fc_dims = kwargs.get('fc_dims', [])

        fc_dims = [in_dim] + fc_dims + [self.rnn_dim]
        self.fc_layers = create_mlp(fc_dims, output_activation=True).to(self.device)

        # recurrent block
        self.rnn_type = kwargs.get('rnn_type', 'linear')  # linear, lstm, gru, myrnn1, myrnn2, myrnn3, elman
        self._get_rnn_layer()

        # readout layer
        head_dims = kwargs.get('head_dims', [])
        head_dims = [self.rnn_dim] + head_dims + [out_dim]
        self.get_logits = create_mlp(head_dims).to(self.device)

        # maximum number of recurrences
        self.max_rec_lim = self.max_rec = kwargs.get('max_rec', 10)

        ### STOPPING CRITERION ###
        self.stop_crit = kwargs.get('stopping_criterion', 'threshold')   # threshold, best, first_correct, learnable

        # criterion arguments
        if self.stop_crit == 'threshold':
            self.threshold = kwargs.get('threshold', .9)
        elif self.stop_crit == 'learnable':
            self.rec_fn_input = kwargs.get('rec_fn_input', 'input') # input, embedding
            rec_fn_type = kwargs.get('rec_fn_type', 'mlp')      # mlp, rbf
            rec_fn_in_dim = in_dim if self.rec_fn_input == 'input' else self.rnn_dim
            if rec_fn_type == 'mlp':
                rec_fn_layers = kwargs.get('rec_fn_layers', [])
                rec_fn_layers = [rec_fn_in_dim] + rec_fn_layers + [self.max_rec]
                self.get_final_step_probs = create_mlp(rec_fn_layers).to(self.device)
            elif rec_fn_type == 'rbf':
                rbf_dim = kwargs.get('rbf_dim', rec_fn_in_dim)
                self.get_final_step_probs = create_rbf(rec_fn_in_dim, rbf_dim, self.max_rec).to(self.device)
            self.tau = kwargs.get('tau', .75)
            self.reg_weight = kwargs.get('reg_weight', 1)
            self.reg_target = kwargs.get('reg_target')
            self.reg_target = torch.tensor(self.reg_target).unsqueeze(0).to(self.device)

            self.learnable_target = kwargs.get('learnable_target', False)

        elif self.stop_crit in ['best', 'first_correct']:
            self.crit_estim = None
            self.use_embedding = False
            self.scaler = None



    def forward(self, input):
        """
        :param input: tensor [batch_size, input_dim]
        :return: output: tensor [batch_size, output_dim]
        """
        if input.dim() == 1:
            input = input.unsqueeze(0)

        a = self.fc_layers(input)

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

        logits_stacked = torch.stack(logits_list)           # [max_rec, batch_size, n_classes]

        return logits_stacked

    def train_step(self, minibatch, optimizer: torch.optim.Optimizer):
        input = minibatch['input']
        target = minibatch['target']

        optimizer.zero_grad()

        logits_stacked = self.forward(input)

        if self.stop_crit == 'learnable':
            loss_stats = self.loss_learnable(input, logits_stacked, target)
        else:
            final_steps = self.get_final_steps(input, logits_stacked, target)

            logits = self.get_final_logits(logits_stacked, final_steps)

            loss_stats = self.loss(logits, target)

        loss_stats['loss'].backward()

        optimizer.step()

        return loss_stats

    def test_step(self, minibatch):
        input = minibatch['input']
        target = minibatch['target']

        logits_stacked = self.forward(input)

        final_steps = self.get_final_steps(input, logits_stacked, target)
        logits = self.get_final_logits(logits_stacked, final_steps)

        loss_stats = self.loss(logits, target)
        metrics = self.metrics(logits_stacked, target, final_steps)

        return loss_stats | metrics

    def get_final_steps(self, input, logits_stacked, target, evaluate=False):
        """
        computes loss
        :param input: input data [batch_size, ]
        :param logits_stacked: logits for each example at each step [max_rec, batch_size, n_classes]
        :param target: target classes [batch_size]
        :return:
        """
        if evaluate and self.stop_crit in ['best', 'first_correct']:
            if self.use_embedding:
                input = self.fc_layers(input)
            input = input.cpu()
            if self.scaler is not None:
                input = self.scaler.transform(input)
            final_steps = self.crit_estim.predict(input)
            final_steps = torch.tensor(final_steps, device=self.device)
            return final_steps

        if self.stop_crit == 'threshold':
            probs = torch.nn.functional.softmax(logits_stacked, dim=-1)  # [max_rec, batch_size, n_classes]
            max_probs, _ = torch.max(probs, dim=-1)                     # [max_rec, batch_size]
            above_thresh_mask = (max_probs > self.threshold).float()  # [max_rec, batch_size]
            above_thresh_mask[-1, :] = 1
            final_steps = torch.argmax(above_thresh_mask, dim=0)        # [batch_size]
        elif self.stop_crit == 'learnable':
            if self.max_rec > 1:
                if self.rec_fn_input == 'embedding':
                    with torch.no_grad():
                        input = self.fc_layers(input)
                final_step_logits = self.get_final_step_probs(input)    # [batch_size, max_rec]
                final_steps = torch.argmax(final_step_logits, dim=1)
            else:
                final_steps = torch.zeros_like(target, device=self.device)
        elif self.stop_crit == 'best':
            target_expanded = target[:, None].expand(-1, self.max_rec)  # [batch_size, max_rec]
            logits_permuted = logits_stacked.permute(1, 2, 0)           # [batch_size, n_classes, max_rec]
            cross_entropy_candidates = nn.functional.cross_entropy(logits_permuted,
                                                                   target_expanded, reduction='none')   # [batch_size, max_rec]
            final_steps = torch.argmin(cross_entropy_candidates, dim=-1)    # [batch_size]
        elif self.stop_crit == 'first_correct':
            final_steps = self.get_first_correct(logits_stacked, target)
        else:
            raise ValueError(f'{self.stop_crit} is not a valid stopping criterion')

        return final_steps

    def get_first_correct(self, logits_stacked, target):
        target_expanded = target[None, :].expand(self.max_rec, -1)  # [max_rec, batch_size]
        predictions = torch.argmax(logits_stacked, dim=-1)  # [max_rec, batch_size]
        correct_mask = (predictions == target_expanded).float()  # [max_rec, batch_size]
        correct_mask[-1, :] = 1
        final_steps = torch.argmax(correct_mask, dim=0)
        return final_steps

    def loss(self, logits, target):
        loss = nn.functional.cross_entropy(logits, target)
        return {'loss': loss, 'cross_entropy': loss}

    def metrics(self, logits_stacked, target, final_steps):
        logits = self.get_final_logits(logits_stacked, final_steps)
        accuracy = self.get_accuracy(logits, target)
        # compute average number of recurrences required
        avg_steps = torch.mean(final_steps.float()) + 1

        return {'accuracy': accuracy,
                'average steps': avg_steps}

    def evaluate(self, minibatch, recurrence=None):
        input = minibatch['input']
        target = minibatch['target']

        logits_stacked = self.forward(input)
        final_steps = self.get_final_steps(input, logits_stacked, target, evaluate=True)
        if recurrence is not None:
            final_steps[:] = recurrence

        logits = self.get_final_logits(logits_stacked, final_steps)

        loss_stats = self.loss(logits, target)
        metrics = self.metrics(logits_stacked, target, final_steps)

        return loss_stats | metrics


    def _get_rnn_layer(self): # linear, lstm, gru, myrnn1, myrnn2, myrnn3, elman
        if self.rnn_type == 'lstm':
            self.rnn_layer = nn.LSTM(input_size=self.rnn_dim,
                                  hidden_size=self.hidden_dim).to(self.device)
        elif self.rnn_type == 'gru':
            self.rnn_layer = nn.GRU(input_size=self.rnn_dim,
                                     hidden_size=self.hidden_dim).to(self.device)
        elif self.rnn_type == 'myrnn1':
            self.rnn_layer = MyRNN(self.rnn_dim, self.hidden_dim, self.device)
        elif self.rnn_type == 'myrnn2':
            self.rnn_layer = MyRNN2(self.rnn_dim, self.device)
        elif self.rnn_type == 'myrnn3':
            self.rnn_layer = MyRNN3(self.rnn_dim, self.device)
        elif self.rnn_type == 'elman':
            self.rnn_layer = ElmanRNN(self.rnn_dim, self.hidden_dim, self.device)
        elif self.rnn_type == 'linear':
            self.rnn_layer = nn.Sequential(nn.Linear(self.rnn_dim, self.rnn_dim),
                                           nn.LeakyReLU()).to(self.device)

    def _get_reg_loss(self, final_step_probs):
        """
        cross entropy between reg target and average final step probs
        """
        if self.reg_weight == 0:
            return 0

        avg_final_step_probs = torch.mean(final_step_probs, dim=0)
        reg_loss = torch.sum((avg_final_step_probs - self.reg_target)**2)
        return reg_loss

    def loss_learnable(self, input, logits_stacked, target):
        if self.learnable_target == 'first_correct':
            final_step_targets = self.get_first_correct(logits_stacked, target)
            if self.max_rec > 1:
                if self.rec_fn_input == 'embedding':
                    with torch.no_grad():
                        input = self.fc_layers(input)
                final_step_logits = self.get_final_step_probs(input)
                learnable_loss = nn.functional.cross_entropy(final_step_logits, final_step_targets)
                final_steps = torch.argmax(final_step_logits, dim=1).detach()
            else:
                learnable_loss = 0
                final_steps = final_step_targets
            logits = self.get_final_logits(logits_stacked, final_steps)
            cross_entropy = nn.functional.cross_entropy(logits, target)

            loss = cross_entropy + learnable_loss
            return {'loss': loss, 'cross_entropy': cross_entropy, 'learnable_loss': learnable_loss}
        else:
            final_step_logits = self.get_final_step_probs(input)    # [batch_size, max_rec]
            final_step_pseudo_sample = nn.functional.gumbel_softmax(final_step_logits, dim=-1, tau=self.tau, hard=True) # [batch_size, max_rec]
            probs_stacked = nn.functional.softmax(logits_stacked, dim=-1)
            weighted_probs = torch.sum(probs_stacked * final_step_pseudo_sample.T.unsqueeze(-1), dim=0)
            mask = (weighted_probs <= 1e-10).float()
            logits = torch.log(weighted_probs + mask * 1e-10)

            cross_entropy = nn.functional.nll_loss(logits, target)

            reg_loss = self._get_reg_loss(final_step_pseudo_sample)

            loss = cross_entropy + self.reg_weight * reg_loss
            return {'loss': loss, 'cross_entropy': cross_entropy, 'reg_loss': reg_loss}

    def get_final_logits(self, logits_stacked, final_steps):
        """
        compute final logits for each example
        :param logits_stacked: logits for each example at each step [max_rec, batch_size, n_classes]
        :param final_steps: indices corresponding to the final step for each example [batch_size]
        :return: logits [batch_size, n_classes]
        """
        # get final logits for each example
        final_steps_exp = final_steps.view(1, -1, 1).expand(-1, -1, logits_stacked.size(-1))  # [1, batch_size, n_classes]
        logits = torch.gather(logits_stacked,
                              index=final_steps_exp,
                              dim=0).squeeze()
        return logits

    def print_layers(self):
        print('model layers:')
        print(self.fc_layers)
        print(self.rnn_layer)
        print(self.get_logits)



def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn