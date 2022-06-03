import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from lab.utils import read_yaml, get_data_loader, get_model

torch.set_grad_enabled(False)

### load the model ###
def load_model(result_dir, model_name, model_version='best_model.pth', device='cuda:0', loader=None):
    model_path = os.path.join(result_dir, model_name, model_version)
    state_dict = torch.load(model_path)

    config_path = os.path.join(result_dir, model_name, 'config.yaml')
    config = read_yaml(config_path)

    torch.manual_seed(config['seed'])

    if loader is None:
        loader = get_data_loader(config)

    model = get_model(config, in_dim=loader.data_dim, out_dim=loader.n_classes)

    model.load_state_dict(state_dict)
    torch.set_grad_enabled(False)

    return model, loader

def evaluate(model, loader):
    acc = 0
    ce = 0
    steps = 0
    counter = 0
    for minibatch in loader.test:
        batch_size = minibatch['target'].size(0)
        stats = model.test_step(minibatch)
        acc += float(stats['accuracy']) * batch_size
        ce += float(stats['cross_entropy']) * batch_size
        steps += float(stats.get('average steps', 0)) * batch_size
        counter += batch_size

    acc = acc / counter * 100
    ce /= counter
    steps /= counter
    return acc, ce, steps

def load_and_evaluate_dir(result_dir, model_dir):
    _, timestamps, _ = next(os.walk(os.path.join(result_dir, model_dir)))
    timestamps = sorted(timestamps)
    loader = None

    acc_list = []
    ce_list = []
    step_list = []
    for timestamp in timestamps:
        model_name = os.path.join(model_dir, timestamp)
        model, loader = load_model(result_dir, model_name, loader=loader)
        loader.make_split()
        acc, ce, steps = evaluate(model, loader)
        acc_list.append(acc)
        ce_list.append(ce)
        step_list.append(steps)

    print(f'accuracy: {np.around(np.mean(acc_list), 2): .2f} +- {np.around(np.std(acc_list), 2): .2f}')
    print(f'cross entropy: {np.around(np.mean(ce_list), 3): .3f} +- {np.around(np.std(ce_list), 3): .3f}')
    if step_list[0] > 0:
        print(f'average steps: {np.around(np.mean(step_list), 3): .3f} +- {np.around(np.std(step_list), 3): .3f}')

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number parameters:', n_params)

    return acc_list
