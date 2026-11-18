"""Bayesian exploration of a finite hyperparameter space with expected improvement."""
from itertools import product

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


def designs(space):
    keys = list(space)
    combinations = list(product(*(space[key] for key in keys)))
    settings = [dict(zip(keys, values)) for values in combinations]
    matrix = np.asarray(combinations, dtype=float)
    matrix = (matrix - matrix.min(0)) / np.maximum(np.ptp(matrix, axis=0), 1e-9)
    return settings, matrix


def choose_next(settings, matrix, visited, seed):
    indices = [settings.index(config) for config, _ in visited]
    target = np.array([score for _, score in visited])
    gp = GaussianProcessRegressor(
        kernel=ConstantKernel(1., (.05, 20.)) * Matern(.55, (.12, 5.), nu=2.5),
        alpha=1e-5, normalize_y=True, random_state=seed)
    gp.fit(matrix[indices], target)
    mean, sd = gp.predict(matrix, return_std=True)
    advantage = mean - target.max() - .0005
    z = advantage / np.maximum(sd, 1e-12)
    improvement = advantage * norm.cdf(z) + sd * norm.pdf(z)
    improvement[indices] = -np.inf
    index = improvement.argmax()
    return settings[index], float(improvement[index])
