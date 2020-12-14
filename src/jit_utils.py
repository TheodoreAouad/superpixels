import numpy as np
from numba import njit


@njit
def numba_mean(ar):
    return np.mean(ar)


@njit
def numba_concatenate(*args, **kwargs):
    return np.concatenate(*args, **kwargs)
