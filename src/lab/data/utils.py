import numpy as np


def make_spiral(n_data, seed=1):
    """
    https://gist.github.com/45deg/e731d9e7f478de134def5668324c44c5
    """
    np.random.seed(seed)

    theta = np.sqrt(np.random.rand(n_data)) * 8 * np.pi

    r_a = 2 * theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    x_a = data_a + np.random.randn(n_data, 2)

    r_b = -2 * theta - np.pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    x_b = data_b + np.random.randn(n_data, 2)

    x = np.concatenate((x_a, x_b), axis=0)
    y = np.concatenate((np.zeros(n_data), np.ones(n_data)), axis=0)

    randperm = np.random.permutation(len(y))
    x = x[randperm]
    y = y[randperm]
    return x, y