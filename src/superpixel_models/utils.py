import numpy as np
from scipy.ndimage import convolve


def compute_edges(mask: np.ndarray) -> np.ndarray:
    ker1 = np.array([[0, 1, -1]])
    ker2 = np.array([[0], [1], [-1]])

    edge1 = convolve(mask, ker1) != 0
    edge2 = convolve(mask, ker2) != 0

    edge1[edge2] = 1

    return edge1
