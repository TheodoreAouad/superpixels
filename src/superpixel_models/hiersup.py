from typing import Optional, List, Tuple, Dict
from functools import total_ordering
from time import time
import heapq

import numpy as np
# from blist import sortedlist

from src.superpixel_models.superpixel import SuperpixelImage


@total_ordering
class WeightItem(tuple):

    def __lt__(self, other):
        return self[0] < other[0]


class Hiersup:

    def __init__(self, max_it: int = None, K: int = None):
        self.max_it = max_it
        self.K_ = K

        self.W_ = None
        self.L_ = None
        self.weights = None
        # self.weights_dict = None
        self.superpixels = None
        self.max_label = None

        self.nb_pop = 0
        self.merged_away = set()

        # debugging
        self.nb_neighbors = []
        self.time_update = []
        self.time_merge = []
        self.time_step = []
        self.time_weights = []
        self.time_tot = []
        self.nb_weights_pop = []

    @staticmethod
    def compute_weights(superpixels: SuperpixelImage, distance) -> List[Tuple[Tuple[int], float]]:
        img = superpixels.array_means
        labels = superpixels.array_label
        W, L = img.shape
        weights = []
        # weights = sortedlist()
        weights_dict = {}
        max_label = 0

        for idx1 in range(W - 1):
            for idx2 in range(L - 1):
                sp1, sp2, sp3 = img[idx1, idx2], img[idx1 + 1, idx2], img[idx1, idx2 + 1]
                lb1, lb2, lb3 = labels[idx1, idx2], labels[idx1 + 1, idx2], labels[idx1, idx2 + 1]

                dist1 = distance(sp1, sp2)
                weights.append(WeightItem(
                    (dist1, (lb1, lb2))
                ))
                # wi = WeightItem(
                #     (dist1, (lb1, lb2))
                # )
                # weights.add(wi)
                # weights_dict[(lb1, lb2)] = wi

                dist2 = distance(sp1, sp3)
                weights.append(WeightItem(
                    (dist2, (lb1, lb3))
                ))
                # wi = WeightItem(
                #     (dist2, (lb1, lb3))
                # )
                # weights.add(wi)
                # weights_dict[(lb1, lb3)] = wi

                max_label = max([max_label, lb1, lb2, lb3])

        # Add last pixels that were not included above.
        for idx2 in range(L - 1):
            sp1, sp2 = img[-1, idx2], img[-1, idx2 + 1]
            lb1, lb2 = labels[-1, idx2], labels[-1, idx2 + 1]
            dist = distance(sp1, sp2)
            weights.append(WeightItem((dist, (lb1, lb2))))

            # wi = WeightItem((dist, (lb1, lb2)))
            # weights.add(wi)
            # weights_dict[(lb1, lb2)] = wi

            max_label = max([max_label, lb1, lb2])

        for idx1 in range(W - 1):
            sp1, sp2 = img[idx1, -1], img[idx1 + 1, -1]
            lb1, lb2 = labels[idx1, -1], labels[idx1 + 1, -1]
            dist = distance(sp1, sp2)
            weights.append(WeightItem((dist, (lb1, lb2))))

            # wi = WeightItem((dist, (lb1, lb2)))
            # weights.add(wi)
            # weights_dict[(lb1, lb2)] = wi

            max_label = max([max_label, lb1, lb2])

        # sp1, sp2 = img[-1, -2], img[-1, -1]
        # lb1, lb2 = labels[-1, -2], labels[-1, -1]
        # dist = distance(sp1, sp2)
        # # weights.append(WeightItem((dist, (lb1, lb2))))
        # wi = WeightItem((dist, (lb1, lb2)))
        # weights.add(wi)
        # weights_dict[(lb1, lb2)] = wi

        heapq.heapify(weights)

        return weights, weights_dict, max_label

    def update_weights(self, new_label: int, ):
        for lb2 in self.superpixels.get_neighbors_of(new_label):
            # t0 = time()
            value = self.distance(
                self.superpixels[new_label].value,
                self.superpixels[lb2].value
            )

            # t1 = time()
            heapq.heappush(self.weights, WeightItem((value, (new_label, lb2))))
            # wi = WeightItem((value, (new_label, lb2)))
            # self.weights.add(wi)
            # self.weights_dict[(new_label, lb2)] = wi
            # t3 = time()

            # print("UPDATE: Computing distance:", t1 - t0)
            # print("UPDATE: Getting index:", t2 - t1)
            # print("UPDATE: Inserting value:", t3 - t2)

    def delete_weights(self, lb: int):
        for lb2 in self.neighbors[lb]:
            key = (lb, lb2) if (lb, lb2) in self.weights_dict.keys() else (lb2, lb)
            self.weights.remove(self.weights_dict[key])
            self.neighbors[lb2].remove(lb)
            del(self.weights_dict[key])


    def merge_verteces(self):
        pass

    @staticmethod
    def distance(c1: float, c2: float) -> float:
        return (c1 - c2) ** 2

    def init_graph(self):
        self.weights, self.weights_dict, self.max_label = self.compute_weights(self.superpixels, self.distance)

    def fit(
        self,
        img: np.ndarray,
        superpixels: Optional[List] = None,
        mask: Optional[np.ndarray] = None,
        verbose: bool = False,
    ):
        self.superpixels = SuperpixelImage(img, superpixels=None, mask=None)  # TODO: add initialisation with some superpixels.
        self.superpixels.compute_neighbors()
        self.init_graph()

        keep = True
        it = 0
        t1 = 0
        t2 = 0
        tot = 0
        while keep:
            print("Step time:", t2 - t1, '\n',
            "Total loop time:", tot, '\n')

            t1 = time()
            self.step()

            t2 = time()
            tot += (t2 - t1)
            self.time_tot.append(tot)
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
        t0 = time()
        keep = True

        nb_weights_pop = 0  # for debugging purpose
        while keep and len(self.weights) > 0:
            dist, (label1, label2) = heapq.heappop(self.weights)
            keep = (label1 in self.merged_away) or (label2 in self.merged_away)
            nb_weights_pop += 1

        self.merged_away.add(label1)
        self.merged_away.add(label2)

        # dist, (label1, label2) = self.weights[0]

        self.max_label += 1
        t1 = time()
        self.superpixels.merge(label1, label2, label_out=self.max_label, track_neighbors=True)


        # self.delete_weights(label1)
        # self.delete_weights(label2)
        t2 = time()


        self.update_weights(self.max_label)
        t3 = time()

        nb_neighbors = len(self.superpixels.get_neighbors_of(self.max_label))
        self.nb_neighbors.append(nb_neighbors)
        self.time_update.append(t3-t2)
        self.time_merge.append(t2-t1)
        self.time_weights.append(t1-t0)
        self.time_step.append(t3-t0)
        self.nb_weights_pop.append(nb_weights_pop)

        print('Nb neighbors', nb_neighbors, '\n',
        'Time to find weight:', t1 - t0, '\n',
        'Time merge:', t2 - t1, '\n',
        'Time update:', t3 - t2, '\n',)
        # print('Total time step ?', t3 - t0)
