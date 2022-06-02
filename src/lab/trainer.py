import os
import shutil
from datetime import datetime

import torch
from tqdm import tqdm
import yaml

from torch.utils.tensorboard import SummaryWriter
from lab.utils import MetricAccumulator, reset_parameters, get_data_loader, get_model

class Trainer:

    def __init__(self, config, **kwargs):
        """
        :param name: str, name of the experiment
        :param model: loaded model that implements train_step and test_step
        :param data_loader: data loader that is iterable
        :param optimizer: torch.optim.Optimizer
        :param kwargs:
            n_epochs: number of epochs to train for
            log_dir: path to the tensorboard logging directory
        """
        name = config['name']
        self.data_loader = get_data_loader(config)
        self.model = get_model(config,
                          in_dim=self.data_loader.data_dim,
                          out_dim=self.data_loader.n_classes)
        self.optimizer = torch.optim.Adam(lr=.001, params=self.model.parameters())

        self.n_epochs = kwargs.get('n_epochs')

        # schedulers
        self.loss_scheduler = kwargs.get('loss_scheduler', None)
        self.max_rec_scheduler = kwargs.get('max_rec_scheduler', None)

        # logging
        timestamp = self.get_timestamp()
        self.log_dir = kwargs.get('log_dir')
        log_dir = os.path.join(self.log_dir, name, timestamp)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.metric_avg = MetricAccumulator()

        # saving
        self.bm_metric = kwargs.get('bm_metric', 'loss')
        self.save_dir = kwargs.get('save_dir')
        self.save_dir = os.path.join(self.save_dir, name, timestamp)
        self.best_metric = None
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_config(config)

        self.early_stop_criterion = kwargs.get('early_stop_criterion', 50)

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
            if self.check_early_stopping():
                break

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
        step_length = self.max_rec_scheduler['step_length']
        if epoch < step_length:
            self.model.max_rec = 1
        elif epoch % step_length == 0:
            self.model.max_rec = min(self.model.max_rec_lim, self.model.max_rec + 1)


    def log(self, metrics, epoch, split='train'):
        for key, value in metrics.items():
            self.writer.add_scalar(f'{split}/{key}', value, epoch)

    def save_model(self, metrics):
        """
        saves the model if it is better according to metric
        """
        metric = metrics[self.bm_metric]
        if self.best_metric is None or metric < self.best_metric:
            self.best_metric = metric
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model.pth'))
            self.early_stop_counter = 0

    def get_timestamp(self):
        dt_obj = datetime.now()
        timestamp = dt_obj.strftime('%m%d-%H%M%S')
        return timestamp

    def save_config(self, config):
        with open(os.path.join(self.save_dir, 'config.yaml'), 'w') as file:
            yaml.dump(config, file)

    def check_early_stopping(self):
        self.early_stop_counter += 1
        if self.early_stop_counter > self.early_stop_criterion:
            self.early_stop_counter = 0
            return True
        else:
            return False



class CrossValidationTrainer(Trainer):
    """
    Trainer for Cross Validation
    Has to be used with a Cross Validation Data Loader
    """

    def __init__(self, config, **kwargs):

        self.data_loader = get_data_loader(config)

        self.n_epochs = kwargs.get('n_epochs')

        # schedulers
        self.loss_scheduler = kwargs.get('loss_scheduler', None)
        self.max_rec_scheduler = kwargs.get('max_rec_scheduler', None)

        self.name = config['name']
        self.log_dir_base = kwargs.get('log_dir')
        self.bm_metric = kwargs.get('bm_metric', 'loss')
        self.save_dir_base = kwargs.get('save_dir')
        self.config = config

        self.early_stop_criterion = kwargs.get('early_stop_criterion', 50)


    def train(self):
        for _ in range(self.data_loader.n_splits):
            self.reset()
            self.data_loader.make_split()
            super().train()

    def reset(self):
        self.model = get_model(self.config,
                               in_dim=self.data_loader.data_dim,
                               out_dim=self.data_loader.n_classes)
        self.optimizer = torch.optim.Adam(lr=.001, params=self.model.parameters())

        # logging
        timestamp = self.get_timestamp()
        log_dir = os.path.join(self.log_dir_base, self.name, timestamp)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.metric_avg = MetricAccumulator()

        # saving
        self.save_dir = os.path.join(self.save_dir_base, self.name, timestamp)
        self.best_metric = None
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_config(self.config)

        self.early_stop_counter = 0