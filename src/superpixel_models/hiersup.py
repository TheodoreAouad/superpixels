from typing import Optional, List, Tuple, Dict
import bisect
from time import time


import numpy as np

from src.superpixel_models.superpixel import SuperpixelImage


class Hiersup:

    def __init__(self, max_it: int = None, K: int = None):
        self.max_it = max_it
        self.K_ = K

        self.W_ = None
        self.L_ = None
        self.weights = None
        self.weights_values = None
        self.superpixels = None
        self.neighbors = None
        self.max_label = None

        self.nb_pop = 0
        self.merged_away = []

    @staticmethod
    def compute_weights(superpixels: SuperpixelImage, distance) -> List[Tuple[Tuple[int], float]]:
        img = superpixels.array_means
        labels = superpixels.array_label
        W, L = img.shape
        weights = []
        weights_values = []
        max_label = 0

        for idx1 in range(W - 1):
            for idx2 in range(L - 1):
                sp1, sp2, sp3 = img[idx1, idx2], img[idx1 + 1, idx2], img[idx1, idx2 + 1]
                lb1, lb2, lb3 = labels[idx1, idx2], labels[idx1 + 1, idx2], labels[idx1, idx2 + 1]

                dist1 = distance(sp1, sp2)
                weights.append(
                    ((lb1, lb2), dist1)
                )
                weights_values.append(dist1)

                dist2 = distance(sp1, sp3)
                weights.append(
                    ((lb1, lb3), dist2)
                )
                weights_values.append(dist2)

                max_label = max([max_label, lb1, lb2, lb3])

        sp1, sp2 = img[-1, -2], img[-1, -1]
        lb1, lb2 = labels[-1, -2], labels[-1, -1]
        dist = distance(sp1, sp2)
        weights.append(((lb1, lb2), dist))
        weights_values.append(dist)
        max_label = max([max_label, lb1, lb2])

        weights.sort(key=lambda x: x[1])
        weights_values.sort()

        return weights, weights_values, max_label

    @staticmethod
    def compute_neighbors(superpixels) -> Dict:
        neighbors = {}
        labels = superpixels.array_label
        W, L = labels.shape

        for idx1 in range(1, W - 1):
            for idx2 in range(1, L - 1):
                neighbors[labels[idx1, idx2]] = set(
                    [labels[idx1 + i, idx2] for i in [-1, 1]] +
                    [labels[idx1, idx2 + i] for i in [-1, 1]]
                )

        for idx1 in range(1, W - 1):
            neighbors[labels[idx1, 0]] = set(
                [labels[idx1 - 1, 0], labels[idx1 + 1, 0], labels[idx1, 1]]
            )
            neighbors[labels[idx1, -1]] = set(
                [labels[idx1 - 1, -1], labels[idx1 + 1, -1], labels[idx1, -2]]
            )

        for idx2 in range(1, L - 1):
            neighbors[labels[0, idx2]] = set(
                [labels[0, idx2 - 1], labels[0, idx2 + 1], labels[1, idx2]]
            )
            neighbors[labels[-1, idx2]] = set(
                [labels[-1, idx2 - 1], labels[-1, idx2 + 1], labels[-2, idx2]]
            )

        neighbors[labels[0, 0]] = set([labels[0, 1], labels[1, 0]])
        neighbors[labels[0, -1]] = set([labels[0, -2], labels[1, -1]])
        neighbors[labels[-1, 0]] = set([labels[-2, 0], labels[-1, 1]])
        neighbors[labels[-1, -1]] = set([labels[-1, -2], labels[-2, -1]])

        return neighbors

    def update_weights(self, new_label: int, ):
        labels = set(
            [(new_label, lb2) for lb2 in self.neighbors[new_label]]
        )

        for lb1, lb2 in labels:
            t0 = time()
            value = self.distance(
                self.superpixels[lb1].value,
                self.superpixels[lb2].value
            )

            t1 = time()
            insert_idx = bisect.bisect(self.weights_values, value)
            t2 = time()
            self.weights.insert(insert_idx, ((lb1, lb2), value))
            self.weights_values.insert(insert_idx, value)
            t3 = time()

            # print("UPDATE: Computing distance:", t1 - t0)
            # print("UPDATE: Getting index:", t2 - t1)
            # print("UPDATE: Inserting value:", t3 - t2)


    def merge_verteces(self):
        pass

    @staticmethod
    def distance(c1: float, c2: float) -> float:
        return (c1 - c2) ** 2

    def init_graph(self):
        self.weights, self.weights_values, self.max_label = self.compute_weights(self.superpixels, self.distance)
        self.neighbors = self.compute_neighbors(self.superpixels)

    def fit(
        self,
        img: np.ndarray,
        superpixels: Optional[List] = None,
        mask: Optional[np.ndarray] = None,
        verbose: bool = False,
    ):
        self.superpixels = SuperpixelImage(img, superpixels=None, mask=None)  # TODO: add initialisation with some superpixels.
        self.init_graph()

        keep = True
        it = 0
        t1 = 0
        while keep:
            print(time() - t1)
            t1 = time()
            self.step()

            it += 1
            keep = (
                ((not self.max_it) or (it < self.max_it)) &
                ((not self.K_) or (len(self.superpixels) > self.K_))
            )
            if verbose:
                if it % 10 == 0:
                    print(it)

        return self

    def step(self):
        keep = True
        while keep and len(self.weights) > 0:
            (label1, label2), dist = self.weights.pop(0)
            self.weights_values.pop(0)
            keep = (label1 in self.merged_away) or (label2 in self.merged_away)

        self.merged_away.append(label1)
        self.merged_away.append(label2)
# 65289
        t1 = time()
        self.max_label += 1
        self.superpixels.merge(label1, label2, label_out=self.max_label)
        t2 = time()
        self.neighbors[self.max_label] = (
            self.neighbors[label1]
            .union(self.neighbors[label2])
            .difference([label1, label2])
        )
        for lb in [label1, label2]:
            for nei in self.neighbors[lb].difference([label1, label2]):
                self.neighbors[nei] = self.neighbors[nei].union([self.max_label]).difference([lb])

        del self.neighbors[label1]
        del self.neighbors[label2]

        self.update_weights(self.max_label)
        t3 = time()

        print('Time merge:', t2 - t1)
        # print('Time update:', t3 - t2)
        # print()