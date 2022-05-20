import click
from pathlib import Path
import torch

from lab.utils import read_yaml, create_instance, get_model, get_trainer, get_dataset, get_data_loader


@click.command()
@click.option('-c', '--config', 'config_path', required=True, type=click.Path(exists=True))


def main(config_path: Path):
    config = read_yaml(config_path)
    name = config['name']
    device = torch.device(config['device'])
    torch.manual_seed(config['seed'])
    print_experiment_info(config)

    print('loading data...')
    if config.get('dataset', None) is not None:
        # if dataset and dataloader are independent
        dataset = get_dataset(config['dataset'], device=device)
    else:
        # if dataset is part of dataloader
        dataset = None

    loader = get_data_loader(config['loader'], dataset)

    print('creating model...')
    model = get_model(config['model'], device=device, in_dim=loader.data_dim, out_dim=loader.n_classes)

    optimizer = torch.optim.Adam(lr=.001, params=model.parameters())

    print('training parameters...')
    trainer = get_trainer(config['trainer'], name, model, loader, optimizer)
    trainer.save_config(config_path)
    trainer.train()


def print_experiment_info(config):
    print('-' * 10,
          '\nexperiment:', config['name'],
          '\nmodel name:', config['model']['name'],
          '\n', '-' * 10)






if __name__ == '__main__':
    main()
