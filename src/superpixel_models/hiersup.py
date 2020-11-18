from itertools import product

from numpy import np

from src.superpixel_models.superpixel import Superpixel


class Hiersup:

    def __init__(self, max_it: int = None, K: int = None):
        self.max_it = max_it
        self.K_ = K

        self.W_ = None
        self.L_ = None
        self.weights = None
        self.super_id = None

    @staticmethod
    def compute_weights(img, distance):
        W, L = img.shape
        weights = [0 for _ in range(int(W*L(W*L-1)/2))]
        all_pixels = list(product(range(W), range(L)))

        idx = 0
        for idx1 in range(W*L):
            for idx2 in range(idx1 + 1, W*L):
                p1, p2 = all_pixels[idx1], all_pixels[idx2]
                weights[idx] = ((p1, p2), distance(p1, p2))
                idx += 1

        weights.sort(key=lambda x: x[1])

        return weights

    def merge_verteces(self):
        pass

    @staticmethod
    def distance(c1, c2):
        return

    def init_graph(self):
        pass

    def fit(self, img):
        self.W_, self.L_ = img.shape
        self.super_id = np.arange(self.W_ * self.L_).reshape(self.W_, self.L_)
        self.weights = self.compute_weights(img, self.distance)

        keep = True
        while keep:
            self.step()

        return

    def step(self):
        v1, v2, dist = self.weights[0]
