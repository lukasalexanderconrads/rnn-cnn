import torch
from torch import nn

from lab.models.base import Model
from lab.models.simple import MLP
from lab.utils import create_mlp, create_instance
from lab.blocks import *

class NLPMLP(Model):
    def __init__(self, in_dim, out_dim: int, vocab_size: int, **kwargs):
        """
        Standard MLP class for natural language
        kwargs:
            device: torch.device
            layer_dims: list of number of neurons in each layer
        """
        super().__init__(**kwargs)
        enc_kwargs = kwargs['encoder']
        self.encoder = create_instance(enc_kwargs['module'], enc_kwargs['name'], enc_kwargs['args'],
                                       self.device, vocab_size)

        # MLP
        dropout = kwargs.get('dropout', 0.0)
        layer_dims = kwargs.get('layer_dims', [])
        layer_dims = [self.encoder.hidden_dim] + layer_dims + [out_dim]

        self.get_logits = create_mlp(layer_dims, dropout=dropout).to(self.device)

    def forward(self, input: torch.Tensor, seq_len: torch.Tensor):
        """
        :param input: tensor [batch_size, input_dim]
        :return: output: tensor [batch_size, output_dim]
        """

        embedding = self.encoder(input, seq_len)
        return self.get_logits(embedding)

    def train_step(self, minibatch, optimizer: torch.optim.Optimizer):
        input = minibatch['input'].to(self.device)
        target = minibatch['target'].to(self.device)
        seq_len = minibatch['length']

        optimizer.zero_grad()

        logits = self.forward(input, seq_len)

        loss = self.loss(logits, target)

        loss.backward()
        optimizer.step()

        return {'loss': loss}

    def test_step(self, minibatch):
        input = minibatch['input'].to(self.device)
        target = minibatch['target'].to(self.device)
        seq_len = minibatch['length']

        logits = self.forward(input, seq_len)

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

class NLPRNN(NLPMLP):
    def __init__(self, in_dim: int, out_dim: int, vocab_size: int, **kwargs):
        """
        MLP with feed forward layers and recurrent layers
        :param kwargs:
            device: torch.device
            fc_dims: list of number of neurons for each layer in feed forward part
            rnn_dim: int, number of neurons of in rnn layer
            out_dim: int, number of output neurons
        """
        super(NLPMLP, self).__init__(**kwargs)

        enc_kwargs = kwargs['encoder']
        self.encoder = create_instance(enc_kwargs['module'], enc_kwargs['name'], enc_kwargs['args'],
                                       self.device, vocab_size)

        # MLP
        dropout = kwargs.get('dropout', 0.0)

        # define network architecture
        head_dims = kwargs.get('head_dims', [])
        self.rnn_dim = self.encoder.hidden_dim

        self.rnn_layer = nn.Sequential(nn.Linear(self.rnn_dim, self.rnn_dim),
                                       nn.LeakyReLU(),
                                       nn.Dropout(dropout)).to(self.device)
        head_dims = [self.rnn_dim] + head_dims + [out_dim]
        self.get_logits = create_mlp(head_dims, dropout=dropout).to(self.device)

        # get recurrency parameters
        self.max_rec_lim = self.max_rec = kwargs.get('max_rec', 10)
        self.threshold = kwargs.get('threshold', .9)
        self.skip_connections = kwargs.get('skip_connections', False)
        self.loss_type = kwargs.get('loss_type', 'final') # first, every, final

    def forward(self, input, seq_len):
        """
        :param input: tensor [batch_size, seq_len]
        :param seq_len: tensor [batch_size]
        :return: output: tensor [batch_size, output_dim]
        """
        h = self.encoder(input, seq_len)

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
        input = minibatch['input'].to(self.device)
        target = minibatch['target'].to(self.device)
        seq_len = minibatch['length']

        optimizer.zero_grad()

        logits_stacked, final_steps = self.forward(input, seq_len)

        loss = self.loss(logits_stacked, target, final_steps)

        loss.backward()
        optimizer.step()

        return {'loss': loss}

    def test_step(self, minibatch):
        input = minibatch['input'].to(self.device)
        target = minibatch['target'].to(self.device)
        seq_len = minibatch['length']

        logits_stacked, final_steps = self.forward(input, seq_len)

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

class RTENLPMLP(NLPMLP):
    def __init__(self, in_dim, out_dim: int, vocab_size: int, **kwargs):
        """
        MLP class for natural language RTE classification
        kwargs:
            device: torch.device
            layer_dims: list of number of neurons in each layer
        """
        super(NLPMLP, self).__init__(**kwargs)
        enc_kwargs = kwargs['encoder']
        self.encoder = create_instance(enc_kwargs['module'], enc_kwargs['name'], enc_kwargs['args'],
                                       self.device, vocab_size)

        # MLP
        dropout = kwargs.get('dropout', 0.0)
        layer_dims = kwargs.get('layer_dims', [])
        layer_dims = [self.encoder.hidden_dim * 2] + layer_dims + [out_dim]

        self.get_logits = create_mlp(layer_dims, dropout=dropout).to(self.device)

        self.vocab = None

    def forward(self, input1, input2, seq_len1, seq_len2):
        """
        :param input: tensor [batch_size, input_dim]
        :return: output: tensor [batch_size, output_dim]
        """
        embedding1 = self.encoder(input1, seq_len1)
        embedding2 = self.encoder(input2, seq_len2)
        embedding = torch.cat((embedding1, embedding2), dim=1)
        return self.get_logits(embedding)

    def train_step(self, minibatch, optimizer: torch.optim.Optimizer):
        input1 = minibatch['input1'].to(self.device)
        input2 = minibatch['input2'].to(self.device)
        target = minibatch['target'].to(self.device)
        seq_len1 = minibatch['length1']
        seq_len2 = minibatch['length2']

        optimizer.zero_grad()

        logits = self.forward(input1, input2, seq_len1, seq_len2)

        loss = self.loss(logits, target)

        loss.backward()
        optimizer.step()

        acc = self.get_accuracy(logits, target)
        return {'loss': loss, 'accuracy': acc}

    def test_step(self, minibatch):
        input1 = minibatch['input1'].to(self.device)
        input2 = minibatch['input2'].to(self.device)
        target = minibatch['target'].to(self.device)
        seq_len1 = minibatch['length1']
        seq_len2 = minibatch['length2']

        #print(self.vocab.lookup_tokens(list(input1[0, :seq_len1[0]])))
        #print(self.vocab.lookup_tokens(list(input2[0, :seq_len2[0]])))
        #print(target[0])
        logits = self.forward(input1, input2, seq_len1, seq_len2)

        metrics = self.metrics(logits, target)

        return metrics

class RTENLPRNN(RTENLPMLP):

    def __init__(self, in_dim, out_dim: int, vocab_size: int, **kwargs):
        """
        MLP class for natural language RTE classification
        kwargs:
            device: torch.device
            layer_dims: list of number of neurons in each layer
        """
        super(NLPMLP, self).__init__(**kwargs)

        enc_kwargs = kwargs['encoder']
        self.encoder = create_instance(enc_kwargs['module'], enc_kwargs['name'], enc_kwargs['args'],
                                       self.device, vocab_size)

        # MLP
        dropout = kwargs.get('dropout', 0.0)

        # define network architecture
        head_dims = kwargs.get('head_dims', [])
        self.rnn_dim = self.encoder.hidden_dim * 2

        self.rnn_layer = nn.Sequential(nn.Linear(self.rnn_dim, self.rnn_dim),
                                       nn.LeakyReLU(),
                                       nn.Dropout(dropout)).to(self.device)
        head_dims = [self.rnn_dim] + head_dims + [out_dim]
        self.get_logits = create_mlp(head_dims, dropout=dropout).to(self.device)

        # get recurrency parameters
        self.max_rec_lim = self.max_rec = kwargs.get('max_rec', 10)
        self.threshold = kwargs.get('threshold', .9)
        self.skip_connections = kwargs.get('skip_connections', False)
        self.loss_type = kwargs.get('loss_type', 'final')  # first, every, final

    def forward(self, input1, input2, seq_len1, seq_len2):
        """
        :param input: tensor [batch_size, input_dim]
        :return: output: tensor [batch_size, output_dim]
        """
        embedding1 = self.encoder(input1, seq_len1)
        embedding2 = self.encoder(input2, seq_len2)
        h = torch.cat((embedding1, embedding2), dim=1)

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
        input1 = minibatch['input1'].to(self.device)
        input2 = minibatch['input2'].to(self.device)
        target = minibatch['target'].to(self.device)
        seq_len1 = minibatch['length1']
        seq_len2 = minibatch['length2']

        optimizer.zero_grad()

        logits_stacked, final_steps = self.forward(input1, input2, seq_len1, seq_len2)

        loss = self.loss(logits_stacked, target, final_steps)

        loss.backward()
        optimizer.step()

        return {'loss': loss}

    def test_step(self, minibatch):
        input1 = minibatch['input1'].to(self.device)
        input2 = minibatch['input2'].to(self.device)
        target = minibatch['target'].to(self.device)
        seq_len1 = minibatch['length1']
        seq_len2 = minibatch['length2']

        logits_stacked, final_steps = self.forward(input1, input2, seq_len1, seq_len2)

        metrics = self.metrics(logits_stacked, target, final_steps)

        return metrics

class Transformer(NLPMLP):

    def __init__(self, in_dim, out_dim: int, vocab_size: int, **kwargs):
        super(NLPMLP, self).__init__(**kwargs)
        self.vocab_size = vocab_size

        self.hidden_dim = kwargs.get('hidden_dim', 64)
        self.n_heads = kwargs.get('n_heads', 2)
        self.n_layers = kwargs.get('n_layers', 2)

        self.dropout = kwargs.get('dropout', 0)

        self.embedding = nn.Embedding(vocab_size + 1, self.hidden_dim, padding_idx=vocab_size).to(self.device)
        self.pos_encoding = PositionalEncoding(d_model=self.hidden_dim, dropout=self.dropout).to(self.device)

        trafo_enc_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.n_heads,
                                                     dim_feedforward=self.hidden_dim * 2, batch_first=True)
        self.trafo = nn.TransformerEncoder(trafo_enc_layer, num_layers=self.n_layers).to(self.device)

        self.get_logits = nn.Sequential(nn.ReLU(), nn.Dropout(self.dropout), nn.Linear(self.hidden_dim, out_dim)).to(self.device)

    def forward(self, input: torch.Tensor, seq_len: torch.Tensor):
        """
        :param input: tensor [batch_size, seq_len]
        :return: output: tensor [batch_size, output_dim]
        """

        x_emb = self.embedding(input)  # [batch_size, seq_len, hidden_dim]
        x_pos_emb = self.pos_encoding(x_emb)

        hidden_states = self.trafo(x_pos_emb)
        pooled_hidden = hidden_states[:, 0]

        return self.get_logits(pooled_hidden)