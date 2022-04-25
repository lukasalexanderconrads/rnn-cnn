import os

from torch.utils.tensorboard import SummaryWriter

class Trainer:

    def __init__(self, name, model, data_loader, optimizer, **kwargs):
        """
        :param name: str, name of the experiment
        :param model: loaded model that implements train_step and test_step
        :param data_loader: data loader that is iterable
        :param optimizer: torch.optim.Optimizer
        :param kwargs:
            n_epochs: number of epochs to train for
            log_dir: path to the tensorboard logging directory
        """
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer

        self.n_epochs = kwargs.get('n_epochs')

        log_dir = kwargs.get('log_dir')
        log_dir = os.path.join(log_dir, name)
        self.writer = SummaryWriter(log_dir=log_dir)

    def train(self):
        for epoch in range(self.n_epochs):
            self.model.train()
            self.train_epoch(epoch)

            self.model.eval()
            self.test_epoch(epoch)

        self.writer.flush()
        self.writer.close()

    def train_epoch(self, epoch):
        for minibatch in self.data_loader.train:
            metrics = self.model.train_step(minibatch, self.optimizer)
            self.writer.add_scalar('train/loss', metrics['loss'], epoch)

    def test_epoch(self, epoch):
        for minibatch in self.data_loader.test:
            metrics = self.model.test_step(minibatch)
            self.writer.add_scalar('test/loss', metrics['loss'], epoch)
            self.writer.add_scalar('test/accuracy', metrics['accuracy'], epoch)





