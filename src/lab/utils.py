import torch
import yaml
from importlib import import_module
from collections import defaultdict


def read_yaml(path):
    """
    read .yaml file and return as dictionary
    :param path: path to .yaml file
    :return: parsed file as dictionary
    """
    with open(path, 'r') as file:
        try:
            parsed_yaml = yaml.load(file, yaml.Loader)
        except yaml.YAMLError as exc:
            print(exc)
    file.close()

    return parsed_yaml

def create_instance(module_name, class_name, kwargs, *args):
    """
    create instance of a class
    :param module_name: str, module the class is in
    :param class_name: str, name of the class
    :param kwargs:
    :return: class instance
    """
    module = import_module(module_name)
    clazz = getattr(module, class_name)
    if kwargs is None:
        instance = clazz(*args)
    else:
        instance = clazz(*args, **kwargs)
    return instance


class MetricAccumulator:
    def __init__(self):
        self.reset()

    def update(self, metrics):
        for key, value in metrics.items():
            self.metrics[key] += value
        self.counter += 1

    def get_average(self):
        for key in self.metrics.keys():
            self.metrics[key] /= self.counter
        return self.metrics

    def reset(self):
        self.metrics = defaultdict(lambda: 0)
        self.counter = 0

def create_mlp(layer_dims, output_activation=False):
    n_layers = len(layer_dims)
    layers = []
    for layer_idx in range(n_layers - 1):
        layers.append(torch.nn.Linear(layer_dims[layer_idx], layer_dims[layer_idx + 1]))
        if layer_idx != n_layers - 2 or output_activation:
            layers.append(torch.nn.LeakyReLU())

    return torch.nn.Sequential(*layers)

def get_trainer(config):
    module_name = config['trainer']['module']
    class_name = config['trainer']['name']
    args = config['trainer']['args']
    trainer = create_instance(module_name, class_name, args, config)
    return trainer


def get_model(config, in_dim, out_dim):
    module_name = config['model']['module']
    class_name = config['model']['name']
    args = config['model']['args']
    args.update({'device': config['device']})
    model = create_instance(module_name, class_name, args, in_dim, out_dim)
    return model


def get_dataset(config, device):
    module_name = config['module']
    class_name = config['name']
    args = config['args']
    dataset = create_instance(module_name, class_name, args, device)
    return dataset


def get_data_loader(config):
    module_name = config['loader']['module']
    class_name = config['loader']['name']
    args = config['loader']['args']
    loader = create_instance(module_name, class_name, args, config['device'])
    return loader

def reset_parameters(model):
    model.apply(lambda w: w.reset_parameters() if hasattr(w, 'reset_parameters') else None)

if __name__ == '__main__':

    parsed_yaml = read_yaml('trainer/test/TEST.yaml')
    print(parsed_yaml)