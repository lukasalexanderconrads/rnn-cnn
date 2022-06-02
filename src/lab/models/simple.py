import torch
from torch import nn

from lab.models.base import Model
from lab.utils import create_mlp

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
        if input.dim() == 1:
            input = input.unsqueeze(0)
        return self.get_logits(input)

    def train_step(self, minibatch, optimizer: torch.optim.Optimizer):
        input = minibatch['input']
        target = minibatch['target']

        optimizer.zero_grad()

        logits = self.forward(input)

        loss = self.loss(logits, target)

        loss.backward()
        optimizer.step()

        return {'loss': loss}

    def test_step(self, minibatch):
        input = minibatch['input']
        target = minibatch['target']

        logits = self.forward(input)

        metrics = self.metrics(logits, target)

        return metrics

    @staticmethod
    def loss(logits, target):
        return nn.functional.cross_entropy(logits, target)

    def metrics(self, logits, target):
        accuracy = self.get_accuracy(logits, target)
        ce_loss = self.loss(logits, target)
        return {'cross_entropy': ce_loss, 'accuracy': accuracy}

    @staticmethod
    def get_accuracy(logits, target):
        prediction = torch.argmax(logits, dim=1)
        accuracy = torch.mean((prediction == target).float())
        return accuracy

    def print_layers(self):
        print('model layers:')
        print(self.get_logits)


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

        # define network architecture
        fc_dims = kwargs.get('fc_dims', [])
        self.rnn_dim = kwargs.get('rnn_dim')
        head_dims = kwargs.get('head_dims', [])

        fc_dims = [in_dim] + fc_dims + [self.rnn_dim]
        self.fc_layers = create_mlp(fc_dims, output_activation=True).to(self.device)

        self.rnn_layer = nn.Sequential(nn.Linear(self.rnn_dim, self.rnn_dim),
                                       nn.LeakyReLU()).to(self.device)
        head_dims = [self.rnn_dim] + head_dims + [out_dim]
        self.get_logits = create_mlp(head_dims).to(self.device)

        # get recurrency parameters
        self.max_rec_lim = self.max_rec = kwargs.get('max_rec', 10)
        self.threshold = kwargs.get('threshold', .9)
        self.skip_connections = kwargs.get('skip_connections', False)
        self.loss_type = kwargs.get('loss_type', 'final') # first, every, final


    def forward(self, input):
        """
        :param input: tensor [batch_size, input_dim]
        :return: output: tensor [batch_size, output_dim]
        """
        if input.dim() == 1:
            input = input.unsqueeze(0)

        h = self.fc_layers(input)
        # store the step index where a logit was above the threshold for the first time
        final_steps = torch.full((input.size(0),), self.max_rec-1).to(self.device)
        logits_list = []
        for step in range(self.max_rec):
            # recurrent layer
            h_new = self.rnn_layer(h)
            h = h + h_new if self.skip_connections else h_new
            # get class probs
            logits = self.get_logits(h)
            logits_list.append(logits)

            # update final_steps
            probs = nn.functional.softmax(logits, dim=1)  # [batch_size, n_classes]
            done_mask = torch.logical_and(torch.max(probs, dim=1).values > self.threshold,
                                          final_steps == self.max_rec-1)
            final_steps[done_mask] = step                   # [batch_size]

            # break if all examples have finished
            if torch.all(final_steps != self.max_rec-1):
                break

        logits_stacked = torch.stack(logits_list)           # [max_rec, batch_size, n_classes]

        return logits_stacked, final_steps

    def train_step(self, minibatch, optimizer: torch.optim.Optimizer):
        input = minibatch['input']
        target = minibatch['target']

        optimizer.zero_grad()

        logits_stacked, final_steps = self.forward(input)

        loss = self.loss(logits_stacked, target, final_steps)

        loss.backward()
        optimizer.step()

        return {'loss': loss}

    def test_step(self, minibatch):
        input = minibatch['input']
        target = minibatch['target']

        logits_stacked, final_steps = self.forward(input)

        loss = self.loss(logits_stacked, target, final_steps)
        metrics = self.metrics(logits_stacked, target, final_steps)

        return {'loss': loss} | metrics

    def loss(self, logits_stacked, target, final_steps):
        """
        computes loss
        :param logits_stacked: logits for each example at each step [max_rec, batch_size, n_classes]
        :param target: target classes [batch_size]
        :param final_steps: indices corresponding to the final step for each example [batch_size]
        :return:
        """
        if self.loss_type == 'first':
            logits = logits_stacked[0]
        elif self.loss_type == 'every':
            logits, target = self.get_valid_logits_target(logits_stacked, target, final_steps)
        elif self.loss_type == 'final':
            logits = self.get_final_logits(logits_stacked, final_steps)

        loss = nn.functional.cross_entropy(logits, target)
        return loss


    def metrics(self, logits_stacked, target, final_steps):
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

    def print_layers(self):
        print('model layers:')
        print(self.fc_layers)
        print(self.rnn_layer)
        print(self.get_logits)

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

    def get_valid_logits_target(self, logits_stacked, target, final_steps):
        """
        flatten logits for steps before the final one and repeat target accordingly
        :param logits_stacked: logits for each example at each step [max_rec, batch_size, n_classes]
        :param target: target classes [batch_size]
        :param final_steps: indices corresponding to the final step for each example [batch_size]
        :return: logits [n_valid_outputs, n_classes], target [n_valid_outputs]
        """
        batch_size = target.size(0)
        n_classes = logits_stacked.size(-1)
        # mask for the valid logits
        arange_exp = torch.arange(0, self.max_rec).unsqueeze(1).expand(-1, batch_size)  # [max_rec, batch_size]
        final_steps_exp = final_steps.unsqueeze(0).expand(self.max_rec, -1)     # [max_rec, batch_size]
        valid_mask = arange_exp <= final_steps_exp              # [max_rec, batch_size]

        valid_indices = valid_mask.flatten().nonzero().squeeze()          # [max_rec * batch_size]
        valid_logits = logits_stacked.view(-1, n_classes)[valid_indices]    # [n_valid_logits, n_classes]
        valid_target = target.repeat_interleave(self.max_rec)[valid_indices]    # [n_valid_logits]

        return valid_logits, valid_target

class LSTM(RNN):
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        """
        MLP with feed forward layers and recurrent layers
        :param kwargs:
        """
        super(MLP, self).__init__(**kwargs)

        # define network architecture
        fc_dims = kwargs.get('fc_dims', [])
        self.lstm_dim = kwargs.get('lstm_dim')
        head_dims = kwargs.get('head_dims', [])


        fc_dims = [in_dim] + fc_dims
        if len(fc_dims) == 0:
            self.fc_layers = lambda x: x
        else:
            self.fc_layers = create_mlp(fc_dims, output_activation=True).to(self.device)

        self.lstm_layer = nn.LSTM(input_size=fc_dims[-1],
                                  hidden_size=self.lstm_dim).to(self.device)

        head_dims = [self.lstm_dim] + head_dims + [out_dim]
        self.get_logits = create_mlp(head_dims).to(self.device)

        # get recurrency parameters
        self.max_rec_lim = self.max_rec = kwargs.get('max_rec', 10)
        self.threshold = kwargs.get('threshold', .9)
        self.skip_connections = kwargs.get('skip_connections', False)
        self.loss_type = kwargs.get('loss_type', 'final')  # first, every, final

    def forward(self, input):
        """
        :param input: tensor [batch_size, input_dim]
        :return: output: tensor [batch_size, output_dim]
        """
        if input.dim() == 1:
            input = input.unsqueeze(0)
        batch_size = input.size(0)

        a = self.fc_layers(input)   # [batch_size, fc_out_dim]
        h = torch.zeros((1, batch_size, self.lstm_dim), device=self.device)
        c = torch.zeros((1, batch_size, self.lstm_dim), device=self.device)
        # store the step index where a logit was above the threshold for the first time
        final_steps = torch.full((input.size(0),), self.max_rec - 1).to(self.device)
        logits_list = []
        for step in range(self.max_rec):
            # recurrent layer
            _, (h, c) = self.lstm_layer(a.unsqueeze(0), (h, c))
            # get class probs
            logits = self.get_logits(h.squeeze())
            logits_list.append(logits)

            # update final_steps
            probs = nn.functional.softmax(logits, dim=1)  # [batch_size, n_classes]
            done_mask = torch.logical_and(torch.max(probs, dim=1).values > self.threshold,
                                          final_steps == self.max_rec - 1)
            final_steps[done_mask] = step  # [batch_size]

            # break if all examples have finished
            if torch.all(final_steps != self.max_rec - 1):
                break

        logits_stacked = torch.stack(logits_list)  # [max_rec, batch_size, n_classes]

        return logits_stacked, final_steps

    def print_layers(self):
        print('model layers:')
        print(self.fc_layers)
        print(self.lstm_layer)
        print(self.get_logits)

class CustomRNN(RNN):
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        """
        MLP with feed forward layers and recurrent layers
        :param kwargs:
        """
        super(MLP, self).__init__(**kwargs)

        self.max_rec_lim = self.max_rec = kwargs.get('max_rec', 10)


        # define network architecture
        fc_dims = kwargs.get('fc_dims', [])
        self.rnn_dim = kwargs.get('rnn_dim')
        head_dims = kwargs.get('head_dims', [])

        fc_dims = [in_dim] + fc_dims + [self.rnn_dim]

        self.fc_layers = create_mlp(fc_dims, output_activation=True).to(self.device)

        self.rnn_layer = nn.Sequential(nn.Linear(self.rnn_dim + self.max_rec_lim, self.rnn_dim),
                                       nn.LeakyReLU()).to(self.device)


        head_dims = [self.rnn_dim] + head_dims + [out_dim]
        self.get_logits = create_mlp(head_dims).to(self.device)

        # get recurrency parameters
        self.threshold = kwargs.get('threshold', .9)
        self.skip_connections = kwargs.get('skip_connections', False)
        self.loss_type = kwargs.get('loss_type', 'final')  # first, every, final

    def forward(self, input):
        """
        :param input: tensor [batch_size, input_dim]
        :return: output: tensor [batch_size, output_dim]
        """
        if input.dim() == 1:
            input = input.unsqueeze(0)
        batch_size = input.size(0)

        h = self.fc_layers(input)  # [batch_size, fc_out_dim]
        onehot_vecs = torch.eye(self.max_rec_lim).to(self.device).unsqueeze(0).expand(batch_size, -1, -1)
        # store the step index where a logit was above the threshold for the first time
        final_steps = torch.full((input.size(0),), self.max_rec - 1).to(self.device)
        logits_list = []
        for step in range(self.max_rec):
            # recurrent layer
            rnn_inp = torch.cat((h, onehot_vecs[:, step]), dim=1)
            h = self.rnn_layer(rnn_inp)
            # get class probs
            logits = self.get_logits(h)
            logits_list.append(logits)

            # update final_steps
            probs = nn.functional.softmax(logits, dim=1)  # [batch_size, n_classes]
            done_mask = torch.logical_and(torch.max(probs, dim=1).values > self.threshold,
                                          final_steps == self.max_rec - 1)
            final_steps[done_mask] = step  # [batch_size]

            # break if all examples have finished
            if torch.all(final_steps != self.max_rec - 1):
                break

        logits_stacked = torch.stack(logits_list)  # [max_rec, batch_size, n_classes]

        return logits_stacked, final_steps

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