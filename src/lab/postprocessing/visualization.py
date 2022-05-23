import torch

def get_grid_points(x_min, x_max, y_min, y_max):
    x1, x2 = torch.meshgrid(torch.arange(x_min, x_max, .05), torch.arange(y_min, y_max, .05))
    points = torch.stack([x1.flatten(), x2.flatten()], dim=1)
    return points