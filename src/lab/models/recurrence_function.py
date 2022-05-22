import torch
from torch import nn

from lab.models.simple import RNN
from lab.utils import create_mlp

class LearnableRNN(RNN):

    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        """
        RNN with learnable function for number of recurrences
        :param kwargs:
        """
        super(LearnableRNN, self).__init__(in_dim, out_dim, **kwargs)

        rec_fn_layers = kwargs.get('rec_fn_layers', [])
        rec_fn_layers = [self.rnn_dim] + rec_fn_layers + [self.max_rec]
        self.get_recurrences = create_mlp(rec_fn_layers).to(self.device)

    def forward(self, input):
        """
        :param input: tensor [batch_size, input_dim]
        :return: output: tensor [batch_size, output_dim]
        """
        if input.dim() == 1:
            input = input.unsqueeze(0)

        a = self.fc_layers(input)
        h = a.clone()
        logits_list = []
        for step in range(self.max_rec):
            # recurrent layer
            h_new = self.rnn_layer(h)
            h = h + h_new if self.skip_connections else h_new
            # get class probs
            logits = self.get_logits(h)
            logits_list.append(logits)

        final_step_logits = self.get_recurrences(input)    # [batch_size, max_rec]

        final_step_probs = nn.functional.gumbel_softmax(final_step_logits, dim=-1, tau=.5, hard=True)

        logits_stacked = torch.stack(logits_list)           # [max_rec, batch_size, n_classes]

        return logits_stacked, final_step_probs

    def train_step(self, minibatch, optimizer: torch.optim.Optimizer):
        input = minibatch['input']
        target = minibatch['target']

        optimizer.zero_grad()

        logits_stacked, final_step_probs = self.forward(input)

        loss = self.loss(logits_stacked, target, final_step_probs)

        loss.backward()
        optimizer.step()

        return {'loss': loss}

    def test_step(self, minibatch):
        input = minibatch['input']
        target = minibatch['target']

        logits_stacked, final_step_probs = self.forward(input)

        loss = self.loss(logits_stacked, target, final_step_probs)
        metrics = self.metrics(logits_stacked, target, final_step_probs)

        return {'loss': loss} | metrics

    def loss(self, logits_stacked, target, final_step_probs):
        """
        computes loss
        :param logits_stacked: logits for each example at each step [max_rec, batch_size, n_classes]
        :param target: target classes [batch_size]
        :param final_step_probs: indices corresponding to the final step for each example [batch_size, max_rec]
        :return:
        """

        probs_stacked = nn.functional.softmax(logits_stacked, dim=-1)
        logits = torch.log(torch.sum(probs_stacked * final_step_probs.T.unsqueeze(-1), dim=0))

        loss = nn.functional.nll_loss(logits, target)
        return loss

    def metrics(self, logits_stacked, target, final_step_probs):
        final_steps = torch.argmax(final_step_probs, dim=-1)

        logits = self.get_final_logits(logits_stacked, final_steps)
        accuracy = self.get_accuracy(logits, target)
        # compute average number of recurrences required
        avg_steps = torch.mean(final_steps.float()) + 1

        ce_loss = nn.functional.cross_entropy(logits, target)

        loss_type = ['first', 'every', 'final'].index(self.loss_type)

        return {'cross_entropy': ce_loss,
                'accuracy': accuracy,
                'average steps': avg_steps,
                'loss_type': loss_type}

    # TODO: implement evaluate function that only computes 1 logit according to rec fn