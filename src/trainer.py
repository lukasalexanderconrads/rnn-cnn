import os
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from src.utils import MetricAccumulator

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

        # logging
        log_dir = kwargs.get('log_dir')
        log_dir = os.path.join(log_dir, name)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.metric_avg = MetricAccumulator()

    def train(self):
        for epoch in tqdm(range(self.n_epochs), desc='epochs'):
            self.model.train()
            self.train_epoch(epoch)

            self.model.eval()
            self.test_epoch(epoch)

        self.writer.flush()
        self.writer.close()

    def train_epoch(self, epoch):
        self.metric_avg.reset()
        for minibatch in tqdm(self.data_loader.train, desc='train set', leave=False):
            metrics = self.model.train_step(minibatch, self.optimizer)
            self.metric_avg.update(metrics)
        metrics = self.metric_avg.get_average()
        self.writer.add_scalar('train/loss', metrics['loss'], epoch)

    def test_epoch(self, epoch):
        self.metric_avg.reset()
        for minibatch in tqdm(self.data_loader.test, desc='test set', leave=False):
            metrics = self.model.test_step(minibatch)
            self.metric_avg.update(metrics)
        metrics = self.metric_avg.get_average()
        self.writer.add_scalar('test/loss', metrics['loss'], epoch)
        self.writer.add_scalar('test/accuracy', metrics['accuracy'], epoch)







