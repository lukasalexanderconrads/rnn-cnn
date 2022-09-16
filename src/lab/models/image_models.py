import torch
from torch import nn
import torch.nn.functional as F
from lab.models.base import Model
from lab.utils import create_mlp
from lab.models.simple import MLP
from lab.blocks import MyRNN


class MLP_image(Model):
    def __init__(self, in_dim, out_dim: int, **kwargs):
        """
        Standard MLP class
        kwargs:
            device: torch.device
            layer_dims: list of number of neurons in each layer
        """
        super().__init__(**kwargs)
        in_dim = 3*32*32
        layer_dims = kwargs.get('layer_dims', [])
        layer_dims = [in_dim] + layer_dims + [out_dim]

        self.get_logits = create_mlp(layer_dims).to(self.device)

        print('model layers:')
        print(self.get_logits)

    def forward(self, input: torch.Tensor):
        """
        :param input: tensor [batch_size, input_dim]
        :return: output: tensor [batch_size, output_dim]
        """
        if input.dim() == 1:
            input = input.unsqueeze(0)
        return self.get_logits(input)

    def train_step(self, minibatch, optimizer: torch.optim.Optimizer):
        input = minibatch[0].to(self.device)
        input = torch.flatten(input, start_dim=1)
        target = minibatch[1].to(self.device)


        optimizer.zero_grad()

        logits = self.forward(input)

        loss = self.loss(logits, target)

        loss.backward()
        optimizer.step()

        return {'loss': loss}

    def test_step(self, minibatch):
        input = minibatch[0].to(self.device)
        input = torch.flatten(input, start_dim=1)
        target = minibatch[1].to(self.device)

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



class RNN_image(MLP_image):
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        """
        MLP with feed forward layers and recurrent layers
        :param kwargs:
            device: torch.device
            fc_dims: list of number of neurons for each layer in feed forward part
            rnn_dim: int, number of neurons of in rnn layer
            out_dim: int, number of output neurons
        """
        super(MLP_image, self).__init__(**kwargs)
        in_dim = 3*32*32
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
        input = minibatch[0].to(self.device)
        input = torch.flatten(input, start_dim=1)
        target = minibatch[1].to(self.device)

        optimizer.zero_grad()

        logits_stacked, final_steps = self.forward(input)

        loss = self.loss(logits_stacked, target, final_steps)

        loss.backward()
        optimizer.step()

        return {'loss': loss}

    def test_step(self, minibatch):
        input = minibatch[0].to(self.device)
        input = torch.flatten(input, start_dim=1)
        target = minibatch[1].to(self.device)

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



class LSTM_image(RNN_image):
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        """
        params= (in_dim+1)x + (x+1)x*8 + last layer: 10*(x+1)
        MLP with feed forward layers and recurrent layers
        :param kwargs:
        """
        super(MLP_image, self).__init__(**kwargs)
        in_dim = 3*32*32
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
        self.max_rec = kwargs.get('max_rec', 10)
        self.threshold = kwargs.get('threshold', .9)
        self.skip_connections = kwargs.get('skip_connections', False)
        self.loss_type = kwargs.get('loss_type', 'final')  # first, every, final

        print('model layers:')
        print(self.fc_layers)
        print(self.lstm_layer)
        print(self.get_logits)

    def forward(self, input):
        """
        :param input: tensor [batch_size, input_dim]
        :return: output: tensor [batch_size, output_dim]
        """
        if input.dim() == 1:
            input = input.unsqueeze(0)
        batch_size = input.size(0)


        a = self.fc_layers(input)  # [batch_size, fc_out_dim]
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

class CNN_image(MLP):
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
        self.conv1 = nn.Conv2d(3, 6, 5).to(self.device)
        self.pool = nn.MaxPool2d(2, 2).to(self.device)
        self.conv2 = nn.Conv2d(6, 32, 5).to(self.device)
        self.fc1 = nn.Linear(32 * 5 * 5, 20).to(self.device)
        self.fc2 = nn.Linear(20, 20).to(self.device)
        self.fc3 = nn.Linear(20, 100).to(self.device)
        # define network architecture
        # fc_dims = kwargs.get('fc_dims', [])


    def forward(self, input):
        """
        :param input: tensor [batch_size, input_dim]
        :return: output: tensor [batch_size, output_dim]
        """
        x = input
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    def train_step(self, minibatch, optimizer: torch.optim.Optimizer):
        input = minibatch[0].to(self.device)
        target = minibatch[1].to(self.device)


        optimizer.zero_grad()

        logits = self.forward(input)

        loss = self.loss(logits, target)

        loss.backward()
        optimizer.step()

        return {'loss': loss}

    def test_step(self, minibatch):
        input = minibatch[0].to(self.device)
        target = minibatch[1].to(self.device)


        logits_stacked = self.forward(input)

        loss = self.loss(logits_stacked, target)
        metrics = self.metrics(logits_stacked, target)

        return {'loss': loss} | metrics

class Resnet_18(MLP):
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        """
        MLP with feed forward layers and recurrent layers
        :param kwargs:
            device: torch.device
            fc_dims: list of number of neurons for each layer in feed forward part
            rnn_dim: int, number of neurons of in rnn layer
            out_dim: int, number of output neurons
        """
        num_class = kwargs.get('num_class')
        super(MLP, self).__init__(**kwargs)
        # self.model_res18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False).to(self.device)

        self.model_res18 = resnet18(10).to(self.device)
        # or any of these variants
        # self.conv1 = nn.Conv2d(3, 6, 5).to(self.device)
        # self.pool = nn.MaxPool2d(2, 2).to(self.device)
        # self.conv2 = nn.Conv2d(6, 32, 5).to(self.device)
        # self.fc1 = nn.Linear(32 * 5 * 5, 100).to(self.device)
        # self.fc2 = nn.Linear(100, 100).to(self.device)
        # self.fc = nn.Linear(10, 10).to(self.device)
        # self.fc2 = nn.Linear(10, num_class).to(self.device)
        # define network architecture
        # fc_dims = kwargs.get('fc_dims', [])

    def forward(self, input):
        """
        :param input: tensor [batch_size, input_dim]
        :return: output: tensor [batch_size, output_dim]
        """
        x = input
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.model_res18(x)
        # x = F.relu(self.model_res18(x))
        # x = F.relu(self.fc(x))
        # x = self.fc2(x)

        return x


    def train_step(self, minibatch, optimizer: torch.optim.Optimizer):
        input = minibatch[0].to(self.device)
        target = minibatch[1].to(self.device)


        optimizer.zero_grad()

        logits = self.forward(input)

        loss = self.loss(logits, target)

        loss.backward()
        optimizer.step()

        return {'loss': loss}

    def test_step(self, minibatch):
        input = minibatch[0].to(self.device)
        target = minibatch[1].to(self.device)


        logits_stacked = self.forward(input)

        loss = self.loss(logits_stacked, target)
        metrics = self.metrics(logits_stacked, target)

        return {'loss': loss} | metrics

class Resnet_18_LSTM(MLP):
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
        # self.model_res18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False).to(self.device)
        num_class = kwargs.get('num_class')

        self.model_res18 = resnet18(num_class).to(self.device)

        # define network architecture
        fc_dims = kwargs.get('fc_dims', [])
        self.lstm_dim = kwargs.get('lstm_dim')
        head_dims = kwargs.get('head_dims', [])
        in_dim = num_class
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
        x = input
        # a = F.relu(self.model_res18(x))
        a = self.model_res18(x)
        batch_size = input.size(0)

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

    def train_step(self, minibatch, optimizer: torch.optim.Optimizer):
        input = minibatch[0].to(self.device)
        # input = torch.flatten(input, start_dim=1)
        target = minibatch[1].to(self.device)

        optimizer.zero_grad()

        logits_stacked, final_steps = self.forward(input)

        loss = self.loss(logits_stacked, target, final_steps)

        loss.backward()
        optimizer.step()

        return {'loss': loss}

    def test_step(self, minibatch):
        input = minibatch[0].to(self.device)
        # input = torch.flatten(input, start_dim=1)
        target = minibatch[1].to(self.device)

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
        final_steps_exp = final_steps.view(1, -1, 1).expand(-1, -1,
                                                            logits_stacked.size(-1))  # [1, batch_size, n_classes]
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
        final_steps_exp = final_steps.unsqueeze(0).expand(self.max_rec, -1)  # [max_rec, batch_size]
        valid_mask = arange_exp <= final_steps_exp  # [max_rec, batch_size]

        valid_indices = valid_mask.flatten().nonzero().squeeze()  # [max_rec * batch_size]
        valid_logits = logits_stacked.view(-1, n_classes)[valid_indices]  # [n_valid_logits, n_classes]
        valid_target = target.repeat_interleave(self.max_rec)[valid_indices]  # [n_valid_logits]

        return valid_logits, valid_target

class Resnet_18_RNN(MLP):
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
        num_class = kwargs.get('num_class')

        self.model_res18 = resnet18(num_class).to(self.device)

        in_dim = num_class
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
        self.loss_type = kwargs.get('loss_type', 'final')  # first, every, final

    def forward(self, input , ret_logits = True):
        """
        :param input: tensor [batch_size, input_dim]
        :return: output: tensor [batch_size, output_dim]
        """
        if input.dim() == 1:
            input = input.unsqueeze(0)
        x = input

        h = F.relu(self.model_res18(x))

        # h = self.fc_layers(input)
        # store the step index where a logit was above the threshold for the first time
        final_steps = torch.full((input.size(0),), self.max_rec - 1).to(self.device)
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
                                          final_steps == self.max_rec - 1)
            final_steps[done_mask] = step  # [batch_size]

            # break if all examples have finished
            if torch.all(final_steps != self.max_rec - 1):
                break

        logits_stacked = torch.stack(logits_list)  # [max_rec, batch_size, n_classes]
        if not ret_logits:
            return logits_stacked, final_steps
        else:
            return self.get_final_logits(logits_stacked, final_steps)

    def train_step(self, minibatch, optimizer: torch.optim.Optimizer):
        input = minibatch[0].to(self.device)
        # input = torch.flatten(input, start_dim=1)
        target = minibatch[1].to(self.device)

        optimizer.zero_grad()

        logits_stacked, final_steps = self.forward(input)

        loss = self.loss(logits_stacked, target, final_steps)

        loss.backward()
        optimizer.step()

        return {'loss': loss}

    def test_step(self, minibatch):
        input = minibatch[0].to(self.device)
        # input = torch.flatten(input, start_dim=1)
        target = minibatch[1].to(self.device)

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
        final_steps_exp = final_steps.view(1, -1, 1).expand(-1, -1,
                                                            logits_stacked.size(-1))  # [1, batch_size, n_classes]
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
        final_steps_exp = final_steps.unsqueeze(0).expand(self.max_rec, -1)  # [max_rec, batch_size]
        valid_mask = arange_exp <= final_steps_exp  # [max_rec, batch_size]

        valid_indices = valid_mask.flatten().nonzero().squeeze()  # [max_rec * batch_size]
        valid_logits = logits_stacked.view(-1, n_classes)[valid_indices]  # [n_valid_logits, n_classes]
        valid_target = target.repeat_interleave(self.max_rec)[valid_indices]  # [n_valid_logits]

        return valid_logits, valid_target

class Resnet_18_hRNN(MLP):
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
        num_class = kwargs.get('num_class')

        self.model_res18 = resnet18(num_class).to(self.device)

        in_dim = num_class
        # define network architecture
        fc_dims = kwargs.get('fc_dims', [])
        self.rnn_dim = kwargs.get('rnn_dim')
        head_dims = kwargs.get('head_dims', [])

        fc_dims = [in_dim] + fc_dims + [self.rnn_dim]
        self.fc_layers = create_mlp(fc_dims, output_activation=True).to(self.device)

        self.hidden_dim = 10
        self.myRNN = MyRNN(in_dim=fc_dims[-1], hidden_dim = self.hidden_dim,device=self.device)
        head_dims = [self.rnn_dim] + head_dims + [out_dim]
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
        x = input

        a = F.relu(self.model_res18(x))

        # h = self.fc_layers(input)
        # store the step index where a logit was above the threshold for the first time
        final_steps = torch.full((input.size(0),), self.max_rec - 1).to(self.device)
        logits_list = []
        h= None
        for step in range(self.max_rec):
            # recurrent layer
            y, h = self.myRNN(a, h)
            # get class probs
            logits = self.get_logits(y)
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

    def train_step(self, minibatch, optimizer: torch.optim.Optimizer):
        input = minibatch[0].to(self.device)
        # input = torch.flatten(input, start_dim=1)
        target = minibatch[1].to(self.device)

        optimizer.zero_grad()

        logits_stacked, final_steps = self.forward(input)

        loss = self.loss(logits_stacked, target, final_steps)

        loss.backward()
        optimizer.step()

        return {'loss': loss}

    def test_step(self, minibatch):
        input = minibatch[0].to(self.device)
        # input = torch.flatten(input, start_dim=1)
        target = minibatch[1].to(self.device)

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
        final_steps_exp = final_steps.view(1, -1, 1).expand(-1, -1,
                                                            logits_stacked.size(-1))  # [1, batch_size, n_classes]
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
        final_steps_exp = final_steps.unsqueeze(0).expand(self.max_rec, -1)  # [max_rec, batch_size]
        valid_mask = arange_exp <= final_steps_exp  # [max_rec, batch_size]

        valid_indices = valid_mask.flatten().nonzero().squeeze()  # [max_rec * batch_size]
        valid_logits = logits_stacked.view(-1, n_classes)[valid_indices]  # [n_valid_logits, n_classes]
        valid_target = target.repeat_interleave(self.max_rec)[valid_indices]  # [n_valid_logits]

        return valid_logits, valid_target
class Resnet_50(MLP):
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        """
        MLP with feed forward layers and recurrent layers
        :param kwargs:
            device: torch.device
            fc_dims: list of number of neurons for each layer in feed forward part
            rnn_dim: int, number of neurons of in rnn layer
            out_dim: int, number of output neurons
        """
        num_class = kwargs.get('num_class')
        super(MLP, self).__init__(**kwargs)
        # self.model_res50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False).to(self.device)

        self.model_res50 = resnet101(1024).to(self.device)
        # or any of these variants
        # self.conv1 = nn.Conv2d(3, 6, 5).to(self.device)
        # self.pool = nn.MaxPool2d(2, 2).to(self.device)
        # self.conv2 = nn.Conv2d(6, 32, 5).to(self.device)
        # self.fc1 = nn.Linear(32 * 5 * 5, 100).to(self.device)
        # self.fc2 = nn.Linear(100, 100).to(self.device)
        self.fc = nn.Linear(1024, num_class).to(self.device)
        # define network architecture
        # fc_dims = kwargs.get('fc_dims', [])

    def forward(self, input):
        """
        :param input: tensor [batch_size, input_dim]
        :return: output: tensor [batch_size, output_dim]
        """
        x = input
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = torch.flatten(x, 1) # flatten all dimensions except batch
        # x = self.model_res50(x)
        x = F.relu(self.model_res50(x))
        x = self.fc(x)
        return x


    def train_step(self, minibatch, optimizer: torch.optim.Optimizer):
        input = minibatch[0].to(self.device)
        target = minibatch[1].to(self.device)


        optimizer.zero_grad()

        logits = self.forward(input)

        loss = self.loss(logits, target)

        loss.backward()
        optimizer.step()

        return {'loss': loss}

    def test_step(self, minibatch):
        input = minibatch[0].to(self.device)
        target = minibatch[1].to(self.device)


        logits_stacked = self.forward(input)

        loss = self.loss(logits_stacked, target)
        metrics = self.metrics(logits_stacked, target)

        return {'loss': loss} | metrics
class CNN_image_MLP(MLP):
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
        self.conv1 = nn.Conv2d(3, 6, 5).to(self.device)
        self.pool = nn.MaxPool2d(2, 2).to(self.device)
        self.conv2 = nn.Conv2d(6, 16, 5).to(self.device)
        self.fc1 = nn.Linear(16 * 5 * 5, 20).to(self.device)
        self.fc2 = nn.Linear(20, 20).to(self.device)
        self.fc3 = nn.Linear(20, 20).to(self.device)
        self.fc4 = nn.Linear(20, 100).to(self.device)
        # define network architecture
        # fc_dims = kwargs.get('fc_dims', [])


    def forward(self, input):
        """
        :param input: tensor [batch_size, input_dim]
        :return: output: tensor [batch_size, output_dim]
        """
        x = input
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


    def train_step(self, minibatch, optimizer: torch.optim.Optimizer):
        input = minibatch[0].to(self.device)
        target = minibatch[1].to(self.device)


        optimizer.zero_grad()

        logits = self.forward(input)

        loss = self.loss(logits, target)

        loss.backward()
        optimizer.step()

        return {'loss': loss}

    def test_step(self, minibatch):
        input = minibatch[0].to(self.device)
        target = minibatch[1].to(self.device)


        logits_stacked = self.forward(input)

        loss = self.loss(logits_stacked, target)
        metrics = self.metrics(logits_stacked, target)

        return {'loss': loss} | metrics


class CNN_image_RNN(MLP):
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
        self.conv1 = nn.Conv2d(3, 6, 5).to(self.device)
        self.pool = nn.MaxPool2d(2, 2).to(self.device)
        self.conv2 = nn.Conv2d(6, 16, 5).to(self.device)
        self.fc1 = nn.Linear(16 * 5 * 5, 20).to(self.device)

        in_dim = 20
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
        self.loss_type = kwargs.get('loss_type', 'final')  # first, every, final

    def forward(self, input):
        """
        :param input: tensor [batch_size, input_dim]
        :return: output: tensor [batch_size, output_dim]
        """
        if input.dim() == 1:
            input = input.unsqueeze(0)
        x = input
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        h = F.relu(self.fc1(x))

        # h = self.fc_layers(input)
        # store the step index where a logit was above the threshold for the first time
        final_steps = torch.full((input.size(0),), self.max_rec - 1).to(self.device)
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
                                          final_steps == self.max_rec - 1)
            final_steps[done_mask] = step  # [batch_size]

            # break if all examples have finished
            if torch.all(final_steps != self.max_rec - 1):
                break

        logits_stacked = torch.stack(logits_list)  # [max_rec, batch_size, n_classes]

        return logits_stacked, final_steps

    def train_step(self, minibatch, optimizer: torch.optim.Optimizer):
        input = minibatch[0].to(self.device)
        # input = torch.flatten(input, start_dim=1)
        target = minibatch[1].to(self.device)

        optimizer.zero_grad()

        logits_stacked, final_steps = self.forward(input)

        loss = self.loss(logits_stacked, target, final_steps)

        loss.backward()
        optimizer.step()

        return {'loss': loss}

    def test_step(self, minibatch):
        input = minibatch[0].to(self.device)
        # input = torch.flatten(input, start_dim=1)
        target = minibatch[1].to(self.device)

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
        final_steps_exp = final_steps.view(1, -1, 1).expand(-1, -1,
                                                            logits_stacked.size(-1))  # [1, batch_size, n_classes]
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
        final_steps_exp = final_steps.unsqueeze(0).expand(self.max_rec, -1)  # [max_rec, batch_size]
        valid_mask = arange_exp <= final_steps_exp  # [max_rec, batch_size]

        valid_indices = valid_mask.flatten().nonzero().squeeze()  # [max_rec * batch_size]
        valid_logits = logits_stacked.view(-1, n_classes)[valid_indices]  # [n_valid_logits, n_classes]
        valid_target = target.repeat_interleave(self.max_rec)[valid_indices]  # [n_valid_logits]

        return valid_logits, valid_target

class CNN_image_myRNN(MLP):

    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        """
        params: (x+h+1)(x+h)
        MLP with feed forward layers and recurrent layers
        :param kwargs:
            device: torch.device
            fc_dims: list of number of neurons for each layer in feed forward part
            rnn_dim: int, number of neurons of in rnn layer
            out_dim: int, number of output neurons
        """
        super(MLP, self).__init__(**kwargs)
        self.conv1 = nn.Conv2d(3, 6, 5).to(self.device)
        self.pool = nn.MaxPool2d(2, 2).to(self.device)
        self.conv2 = nn.Conv2d(6, 16, 5).to(self.device)
        self.fc1 = nn.Linear(16 * 5 * 5, 20).to(self.device)

        in_dim = 20
        # define network architecture
        fc_dims = kwargs.get('fc_dims', [])
        self.rnn_dim = kwargs.get('rnn_dim')
        head_dims = kwargs.get('head_dims', [])

        fc_dims = [in_dim] + fc_dims + [self.rnn_dim]
        self.fc_layers = create_mlp(fc_dims, output_activation=True).to(self.device)

        self.hidden_dim = 9
        self.myRNN = MyRNN(in_dim=fc_dims[-1], hidden_dim = self.hidden_dim,device=self.device)
        head_dims = [self.rnn_dim] + head_dims + [out_dim]
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
        x = input
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        a = F.relu(self.fc1(x))
        # a = self.fc_layers(input)  # [batch_size, fc_out_dim]
        # h = torch.zeros((1, batch_size, self.rnn_dim), device=self.device)
        # c = torch.zeros((1, batch_size, self.lstm_dim), device=self.device)
        # store the step index where a logit was above the threshold for the first time
        final_steps = torch.full((input.size(0),), self.max_rec - 1).to(self.device)
        logits_list = []
        h = None
        for step in range(self.max_rec):
            # recurrent layer
            y,h= self.myRNN(a, h)
            # get class probs
            logits = self.get_logits(y)
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

    def train_step(self, minibatch, optimizer: torch.optim.Optimizer):
        input = minibatch[0].to(self.device)
        # input = torch.flatten(input, start_dim=1)
        target = minibatch[1].to(self.device)

        optimizer.zero_grad()

        logits_stacked, final_steps = self.forward(input)

        loss = self.loss(logits_stacked, target, final_steps)

        loss.backward()
        optimizer.step()

        return {'loss': loss}

    def test_step(self, minibatch):
        input = minibatch[0].to(self.device)
        # input = torch.flatten(input, start_dim=1)
        target = minibatch[1].to(self.device)

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
        final_steps_exp = final_steps.view(1, -1, 1).expand(-1, -1,
                                                            logits_stacked.size(-1))  # [1, batch_size, n_classes]
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
        final_steps_exp = final_steps.unsqueeze(0).expand(self.max_rec, -1)  # [max_rec, batch_size]
        valid_mask = arange_exp <= final_steps_exp  # [max_rec, batch_size]

        valid_indices = valid_mask.flatten().nonzero().squeeze()  # [max_rec * batch_size]
        valid_logits = logits_stacked.view(-1, n_classes)[valid_indices]  # [n_valid_logits, n_classes]
        valid_target = target.repeat_interleave(self.max_rec)[valid_indices]  # [n_valid_logits]

        return valid_logits, valid_target

class CNN_image_LSTM(MLP):
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
        self.conv1 = nn.Conv2d(3, 6, 5).to(self.device)
        self.pool = nn.MaxPool2d(2, 2).to(self.device)
        self.conv2 = nn.Conv2d(6, 16, 5).to(self.device)
        self.fc1 = nn.Linear(16 * 5 * 5, 20).to(self.device)

        in_dim = 20
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
        x = input
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        a = F.relu(self.fc1(x))

        batch_size = input.size(0)

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

    def train_step(self, minibatch, optimizer: torch.optim.Optimizer):
        input = minibatch[0].to(self.device)
        # input = torch.flatten(input, start_dim=1)
        target = minibatch[1].to(self.device)

        optimizer.zero_grad()

        logits_stacked, final_steps = self.forward(input)

        loss = self.loss(logits_stacked, target, final_steps)

        loss.backward()
        optimizer.step()

        return {'loss': loss}

    def test_step(self, minibatch):
        input = minibatch[0].to(self.device)
        # input = torch.flatten(input, start_dim=1)
        target = minibatch[1].to(self.device)

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
        final_steps_exp = final_steps.view(1, -1, 1).expand(-1, -1,
                                                            logits_stacked.size(-1))  # [1, batch_size, n_classes]
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
        final_steps_exp = final_steps.unsqueeze(0).expand(self.max_rec, -1)  # [max_rec, batch_size]
        valid_mask = arange_exp <= final_steps_exp  # [max_rec, batch_size]

        valid_indices = valid_mask.flatten().nonzero().squeeze()  # [max_rec * batch_size]
        valid_logits = logits_stacked.view(-1, n_classes)[valid_indices]  # [n_valid_logits, n_classes]
        valid_target = target.repeat_interleave(self.max_rec)[valid_indices]  # [n_valid_logits]

        return valid_logits, valid_target

class MY_RNN_image(RNN_image):
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        """
        MLP with feed forward layers and recurrent layers
        :param kwargs:
        """
        super(MLP_image, self).__init__(**kwargs)
        in_dim = 3*32*32
        # define network architecture
        fc_dims = kwargs.get('fc_dims', [])
        self.rnn_dim = kwargs.get('rnn_dim')
        head_dims = kwargs.get('head_dims', [])

        fc_dims = [in_dim] + fc_dims + [self.rnn_dim]
        self.fc_layers = create_mlp(fc_dims, output_activation=True).to(self.device)
        self.hidden_dim = 8
        self.myRNN = MyRNN(in_dim=fc_dims[-1], hidden_dim = self.hidden_dim,device=self.device)

        head_dims = [self.rnn_dim] + head_dims + [out_dim]
        self.get_logits = create_mlp(head_dims).to(self.device)

        # get recurrency parameters
        self.max_rec = kwargs.get('max_rec', 10)
        self.threshold = kwargs.get('threshold', .9)
        self.skip_connections = kwargs.get('skip_connections', False)
        self.loss_type = kwargs.get('loss_type', 'final')  # first, every, final

        print('model layers:')
        print(self.fc_layers)
        # print(self.lstm_layer)
        print(self.get_logits)

    def forward(self, input):
        """
        :param input: tensor [batch_size, input_dim]
        :return: output: tensor [batch_size, output_dim]
        """
        if input.dim() == 1:
            input = input.unsqueeze(0)
        batch_size = input.size(0)

        a = self.fc_layers(input)  # [batch_size, fc_out_dim]
        # h = torch.zeros((1, batch_size, self.rnn_dim), device=self.device)
        # c = torch.zeros((1, batch_size, self.lstm_dim), device=self.device)
        # store the step index where a logit was above the threshold for the first time
        final_steps = torch.full((input.size(0),), self.max_rec - 1).to(self.device)
        logits_list = []
        h = None
        for step in range(self.max_rec):
            # recurrent layer
            y,h= self.myRNN(a, h)
            # get class probs
            logits = self.get_logits(y)
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

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.LeakyReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 2
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.LeakyReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = self.dropout(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

def resnet18(num_classes):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2],num_classes)

def resnet34(num_classes):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3],num_classes)

def resnet50(num_classes):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3],num_classes)

def resnet101(num_classes):
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3],num_classes)

def resnet152(num_classes):
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3],num_classes)