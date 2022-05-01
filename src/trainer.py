import os
from tqdm import tqdm
from datetime import datetime

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

        self.loss_scheduler = kwargs.get('loss_scheduler', None)
        self.max_rec_scheduler = kwargs.get('max_rec_scheduler', None)

        # logging
        timestamp = self.get_timestamp()
        log_dir = kwargs.get('log_dir')
        log_dir = os.path.join(log_dir, name, timestamp)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.metric_avg = MetricAccumulator()

    def train(self):
        for epoch in tqdm(range(self.n_epochs), desc='epochs'):
            if self.loss_scheduler is not None:
                self.update_loss_type(epoch)
            if self.max_rec_scheduler is not None:
                self.update_max_rec(epoch)
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
        self.log(metrics, epoch)

    def test_epoch(self, epoch):
        self.metric_avg.reset()
        for minibatch in tqdm(self.data_loader.test, desc='test set', leave=False):
            metrics = self.model.test_step(minibatch)
            self.metric_avg.update(metrics)
        metrics = self.metric_avg.get_average()
        self.log(metrics, epoch, 'test')

    def update_loss_type(self, epoch):
        epoch_frac = epoch / self.n_epochs
        if epoch_frac < self.loss_scheduler['first_step']:
            self.model.loss_type = 'first'
        elif epoch_frac < self.loss_scheduler['every_step']:
            self.model.loss_type = 'every'
        else:
            self.model.loss_type = 'final'

    def update_max_rec(self, epoch):
        min = self.max_rec_scheduler['min']
        max = self.max_rec_scheduler['max']
        epoch_frac = epoch / self.n_epochs
        max_rec = int(min + epoch_frac * (max - min))
        self.model.max_rec = max_rec

    def log(self, metrics, epoch, split='train'):
        for key, value in metrics.items():
            self.writer.add_scalar(f'{split}/{key}', value, epoch)

    def get_timestamp(self):
        dt_obj = datetime.now()
        timestamp = f'{dt_obj.month}{dt_obj.day}-{dt_obj.hour}{dt_obj.minute}{dt_obj.second}'
        return timestamp






