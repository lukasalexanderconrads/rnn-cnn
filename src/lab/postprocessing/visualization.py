import torch
import matplotlib.pyplot as plt

def get_grid_points(x_min, x_max, y_min, y_max, step_size=.05):
    x1, x2 = torch.meshgrid(torch.arange(x_min, x_max, step_size), torch.arange(y_min, y_max, step_size))
    points = torch.stack([x1.flatten(), x2.flatten()], dim=1)
    return points

def make_violin_plot(population_list):
    plt.violinplot(population_list, range(len(population_list)), points=20, widths=0.3,
                   showmeans=True, showextrema=True, showmedians=False)
    for i, pop in enumerate(population_list):
        plt.scatter([i]*len(pop), pop, c='red', alpha=.6)