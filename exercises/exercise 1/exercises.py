import numpy as np
from numba import jit


def linear_regression(x, y, ϵ=None, order=1):
    # Set up the design matrix
    X = np.ones_like(x)
    for n in range(1, order+1):
        X = np.append(X, x**n, axis=1)
    β = _lin_reg(X, y)
    return β


@jit(nopython=True)
def _lin_reg(X, y):
    return np.linalg.inv(X.T@X)@X.T@y
