import torch
import matplotlib.pyplot as plt
import os
from lab.utils import read_yaml, get_data_loader, get_model, get_dataset

torch.set_grad_enabled(False)

### load the model ###
def load_model(result_dir, model_name, model_version='best_model.pth', device='cuda:0'):
    model_path = os.path.join(result_dir, model_name, model_version)
    state_dict = torch.load(model_path)

    config_path = os.path.join(result_dir, model_name, 'config.yaml')
    config = read_yaml(config_path)

    torch.manual_seed(config['seed'])

    loader = get_data_loader(config['loader'], None)

    model = get_model(config['model'], device=device, in_dim=loader.data_dim, out_dim=loader.n_classes)

    model.load_state_dict(state_dict)

    return model, loader





