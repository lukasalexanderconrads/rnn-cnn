import os
import shutil
from datetime import datetime

import torch
from tqdm import tqdm
import yaml

from torch.utils.tensorboard import SummaryWriter
from lab.utils import MetricAccumulator

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

        # schedulers
        self.loss_scheduler = kwargs.get('loss_scheduler', None)
        self.max_rec_scheduler = kwargs.get('max_rec_scheduler', None)

        # logging
        timestamp = self.get_timestamp()
        log_dir = kwargs.get('log_dir')
        log_dir = os.path.join(log_dir, name, timestamp)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.metric_avg = MetricAccumulator()

        # saving
        self.bm_metric = kwargs.get('bm_metric', 'loss')
        self.save_dir = kwargs.get('save_dir')
        self.save_dir = os.path.join(self.save_dir, name, timestamp)
        self.best_metric = None
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

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
        self.save_model(metrics)

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
        max_rec = round(min + epoch_frac * (max - min))
        self.model.max_rec = max_rec

    def log(self, metrics, epoch, split='train'):
        for key, value in metrics.items():
            self.writer.add_scalar(f'{split}/{key}', value, epoch)

    def save_model(self, metrics):
        """
        saves the model if it is better according to metric
        """
        metric = metrics[self.bm_metric]
        if self.best_metric is None or self.best_metric < metric:
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model.pth'))

    def get_timestamp(self):
        dt_obj = datetime.now()
        timestamp = dt_obj.strftime('%m%d-%H%M%S')
        return timestamp

    def save_config(self, config):
        with open(os.path.join(self.save_dir, 'config.yaml'), 'w') as file:
            yaml.dump(config, file)



