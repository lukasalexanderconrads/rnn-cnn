import click
from pathlib import Path
import torch

from src.utils import read_yaml, create_instance


@click.command()
@click.option('-c', '--config', 'config_path', required=True, type=click.Path(exists=True))


def main(config_path: Path):
    config = read_yaml(config_path)
    name = config['name']
    device = torch.device(config['device'])
    print_experiment_info(config)

    dataset = get_dataset(config['dataset'], device=device)


    print('creating model...')
    model = get_model(config['model'], device=device, in_dim=dataset.data_dim, out_dim=dataset.n_classes)

    print('loading data...')
    loader = get_data_loader(config['loader'], dataset)

    optimizer = torch.optim.Adam(lr=.001, params=model.parameters())

    print('training parameters...')
    trainer = get_trainer(config['trainer'], name, model, loader, optimizer)

    trainer.train()


def get_trainer(config, name, model, loader, optimizer):
    module_name = config['module']
    class_name = config['name']
    args = config['args']
    trainer = create_instance(module_name, class_name, args, name, model, loader, optimizer)
    return trainer


def get_model(config, device, in_dim, out_dim):
    module_name = config['module']
    class_name = config['name']
    args = config['args']
    args.update({'device': device})
    model = create_instance(module_name, class_name, args, in_dim, out_dim)
    return model


def get_dataset(config, device):
    module_name = config['module']
    class_name = config['name']
    args = config['args']
    dataset = create_instance(module_name, class_name, args, device)
    return dataset


def get_data_loader(config, dataset):
    module_name = config['module']
    class_name = config['name']
    args = config['args']
    loader = create_instance(module_name, class_name, args, dataset)
    return loader

def print_experiment_info(config):
    print('-' * 10,
          '\nexperiment:', config['name'],
          '\nmodel name:', config['model']['name'],
          '\ndata set:', config['dataset']['name'],
          '\n', '-' * 10)






if __name__ == '__main__':
    main()
