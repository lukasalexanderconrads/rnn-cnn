import click
from pathlib import Path
import torch

from lab.utils import read_yaml, create_instance, get_model, get_trainer, get_dataset, get_data_loader
from lab.expand_config import expand_config


@click.command()
@click.option('-c', '--config', 'config_path', required=True, type=click.Path(exists=True))


def main(config_path: Path):
    configs = read_yaml(config_path)
    torch.manual_seed(configs['seed'])

    config_list = expand_config(configs)

    for config in config_list:
        name = config['name']

        device = torch.device(config['device'])

        print_experiment_info(config)

        print('loading data...')
        loader = get_data_loader(config['loader'], device)

        print('creating model...')
        model = get_model(config['model'], device=device, in_dim=loader.data_dim, out_dim=loader.n_classes)

        optimizer = torch.optim.Adam(lr=.001, params=model.parameters())

        print('training parameters...')
        trainer = get_trainer(config['trainer'], name, model, loader, optimizer)
        trainer.save_config(config)
        trainer.train()


def print_experiment_info(config):
    print('-' * 10,
          '\nexperiment:', config['name'],
          '\nmodel name:', config['model']['name'],
          '\n', '-' * 10)






if __name__ == '__main__':
    main()
