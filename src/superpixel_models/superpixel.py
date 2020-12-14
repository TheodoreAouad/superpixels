from __future__ import annotations
import functools
from typing import List, Optional, Dict, Set

import numpy as np

from src.jit_utils import numba_mean
from src.superpixel_models.utils import compute_edges


def record_state(value_att, state_att):
    """ Decorator that records superpixels state. Recompute the function
    if and only if superpixels did change.
    """

    def decorator_record(fn):

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            class_state = getattr(args[0], state_att)

            if class_state != getattr(wrapper, state_att):
                # wrapper.superpixels_state = args[0].superpixels_state
                setattr(wrapper, state_att, class_state)
                res = fn(*args, **kwargs)
                setattr(args[0], value_att, res)
                return res
            else:
                return getattr(args[0], value_att)

        setattr(wrapper, state_att, -1)


        return wrapper
    return decorator_record


class Superpixel:

    def __init__(
        self,
        label: Optional[int] = None,
        pixel_idxs: Optional[np.ndarray] = None,
        values: Optional[np.ndarray] = None,
    ):
        self._to_reset_on_change = [
            "_value",
            "_pos",
            "_test",
        ]

        self.label = label
        self.pixel_idxs = pixel_idxs
        self.values = values

        self.this_state = 0
        self.superpixels_state = 0

        self._value = None
        self._pos = None
        self._test = None


    @property
    # @record_state(value_att="_value", state_att="this_state")
    def value(self):
        if self._value is not None:
            return self._value

        self._value = numba_mean(self.values)
        return self._value

    @property
    # @record_state(value_att="_pos", state_att="this_state")
    def pos(self):
        if self._pos is not None:
            return self._pos

        self._pos = np.mean(self.pixel_idxs, 0)
        # return np.mean(self.pixel_idxs, 0)
        return self._pos

    def __getitem__(self, idx):
        return self.pixel_idxs[idx], self.values[idx]

    def __len__(self):
        return len(self.pixel_idxs)

    def merge(self, sp: Superpixel, label: Optional[int] = None) -> Superpixel:
        return Superpixel(
            label=self.label if label is None else label,
            # pixel_idxs=self.pixel_idxs + sp.pixel_idxs,
            pixel_idxs=np.concatenate((self.pixel_idxs, sp.pixel_idxs)),
            # values=self.values + sp.values,
            values=np.concatenate((self.values, sp.values)),
        )

    def __repr__(self):
        return (
            f"Superpixel [\n"
            f"   Label: {self.label}\n"
            f"   Pixels: {self.pixel_idxs}\n"
            f"   Values: {self.values}\n"
            "]"
        )

    def reset_attributes(self):
        for attr in self._to_reset_on_change:
            setattr(self, attr, None)
        return self


class SuperpixelImage:

    def __init__(
        self,
        img: np.ndarray,
        superpixels: Optional[Dict[Superpixel]] = None,
        mask: Optional[np.ndarray] = None,
        track_neighbors: Optional[bool] = True,
    ):
        self._to_reset_on_change = [
            "_array_label", "_array_means", "_edges"
        ]

        self.img_ini = img + 0
        self.superpixels_state = 0

        if superpixels is not None:
            self.superpixels = superpixels
        elif mask is not None:
            self.superpixels = self.init_sp_from_mask(img, mask)
        else:
            self.superpixels = self.init_sp_from_img(img)

        self.superpixels_values = list(self.superpixels.values())

        self._array_label = None
        self._array_means = None
        self._neighbors = None
        self._edges = None

        if track_neighbors:
            self.compute_neighbors()

    def __getitem__(self, idx):
        return self.superpixels[idx]

    def __len__(self):
        return len(self.superpixels)

    @property
    # @record_state(value_att="_array_label", state_att='superpixels_state')
    def array_label(self) -> np.ndarray:
        if self._array_label is not None:
            return self._array_label

        img = np.zeros(self.img_ini.shape).astype(int)

        for sp in self.superpixels.values():
            for (idx1, idx2), _ in sp:
                img[idx1, idx2] = sp.label
        self._array_label = img
        return self._array_label
        # return img

    @property
    @record_state(value_att="_array_means", state_att='superpixels_state')
    def array_means(self) -> np.ndarray:
        if self._array_means is not None:
            return self._array_means

        img = np.zeros(self.img_ini.shape)

        for sp in self.superpixels.values():
            value = sp.value
            for (idx1, idx2), pixel_id in sp:
                img[idx1, idx2] = value
        # return img
        self._array_means = img
        return self._array_means

    @staticmethod
    def init_sp_from_img(img: np.ndarray) -> List[Superpixel]:
        W, L = img.shape
        superpixels = {}

        idx = 0
        for i in range(W):
            for j in range(L):
                superpixels[idx] = Superpixel(
                    label=idx,
                    pixel_idxs=np.array([[i, j]]),
                    values=np.array([img[i, j]]),
                )
                idx += 1

        return superpixels

    @staticmethod
    def init_sp_from_mask(img: np.ndarray, mask: np.ndarray) -> List[Superpixel]:
        W, L = mask.shape
        superpixels = {}

        idx = 0
        for value in np.unique(mask):
            superpixels[idx] = Superpixel(
                label=idx,
                pixel_idxs=list(zip(*np.where(mask==value))),
                values=list(img[np.where(mask==value)])
            )
            idx += 1

        return superpixels

    # @record_state(value_att="_edges", state_att='superpixels_state')
    def infer_superpixel_edges(self) -> np.ndarray:
        if self._edges is not None:
            return self._edges
        self._edges = compute_edges(self.array_label)
        # return compute_edges(self.array_label)
        return self._edges

    def __repr__(self):
        return super().__repr__().replace('>', f' ({len(self)} superpixels)>')


    def merge(self, label1: int, label2: int, label_out: Optional[int] = None, track_neighbors=True):

        if label2 not in self.get_neighbors_of(label1):
            assert False, "{label2} not in {label1} neighbors"

        if label_out is None:
            label_out = label1

        sp1 = self.superpixels.pop(label1)
        sp2 = self.superpixels.pop(label2)
        self.set_superpixel(label_out, sp1.merge(sp2, label=label_out))


        if track_neighbors:
            self.set_neighbors_of(label_out,
                self.get_neighbors_of(label1)
                .union(self.get_neighbors_of(label2))
                .difference([label1, label2])
            )

            to_go = set([label1, label2]).difference([label_out])

            for lb in [label1, label2]:
                for nei in self.get_neighbors_of(lb).difference(to_go):
                    self.set_neighbors_of(
                        nei,
                        self.get_neighbors_of(nei).union([label_out]).difference(to_go)
                    )

            for lb in to_go:
                self.remove_from_neighbors(lb)

    # @record_state(value_att="_neighbors", state_att='superpixels_state')
    def compute_neighbors(self):
        # if self._neighbors is not None:
        #     return self._neighbors

        edges = self.infer_superpixel_edges()
        labels = self.array_label
        _neighbors = {}

        xs, ys = np.where(edges)
        for (x, y) in zip(xs, ys):
            lb1 = labels[x, y]
            # lb2 = labels[max(0, x - 1), y]
            lb2 = labels[(x - 1) if (x - 1) > 0 else 0, y]
            # lb3 = labels[x, max(0, y - 1)]
            lb3 = labels[x, (y - 1) if (y - 1) > 0 else 0]

            if lb1 != lb2:
                _neighbors.setdefault(lb1, set()).add(lb2)
                _neighbors.setdefault(lb2, set()).add(lb1)
            if lb1 != lb3:
                _neighbors.setdefault(lb1, set()).add(lb3)
                _neighbors.setdefault(lb3, set()).add(lb1)

        # return _neighbors
        self._neighbors = _neighbors
        return self._neighbors


    def get_neighbors_of(self, label: int) -> List[int]:
        return self._neighbors[label]

    def set_neighbors_of(self, label: int, neis: Set[int]) -> Dict:
        self._neighbors[label] = neis
        return self._neighbors

    def remove_from_neighbors(self, label: int) -> Dict:
        del(self._neighbors[label])
        return self._neighbors

    def update_neighbors(self, labels: List[int]) -> Dict:
        array_label = self.array_label
        for lb in labels:
            new_neis = set()
            if lb not in self.superpixels.keys():
                del(self._neighbors[lb])
                continue

            all_edges = self.infer_superpixel_edges()
            cur_edges = compute_edges(array_label == lb)

            xs, ys = np.where(all_edges * cur_edges)
            for (x, y) in zip(xs, ys):

                lb2 = array_label[max(0, x - 1), y]
                lb3 = array_label[x, max(0, y - 1)]

                if lb2 != lb:
                    new_neis.add(lb2)
                    self._neighbors.setdefault(lb2, set()).add(lb)
                if lb3 != lb:
                    new_neis.add(lb3)
                    self._neighbors.setdefault(lb3, set()).add(lb)

            self.set_neighbors_of(lb, new_neis)

        return self._neighbors



    def array_some_sp(self, labels: List[int]):
        array_label = self.array_label + 0
        array_label[~np.isin(array_label, labels)] = -1
        return array_label


    def reset_attributes(self):
        for attr in self._to_reset_on_change:
            setattr(self, attr, None)
        return self

    def set_superpixel(self, label: int, new_sp: Superpixel) -> None:
        self.superpixels[label] = new_sp
        # self.superpixels_state += 1
        self.reset_attributes()

    def __setattr__(self, name, value):
        if name == "superpixels":
            # self.superpixels_state += 1
            self.reset_attributes()
        return super().__setattr__(name, value)


    def absorb_small(self, size: int, track_neighbors=True, mode='random'):
        changed = {}
        for label, sp in list(self.superpixels.items()):
            if len(sp) <= size:
                neis = self.get_neighbors_of(label)

                if mode == 'random':
                    lb1 = neis.pop()

                elif mode == 'min_pos':
                    min_pos = np.infty
                    min_nei = 0
                    for nei in neis:
                        cur_dist = (self.superpixels[nei].pos - sp.pos) ** 2
                        if cur_dist < min_pos:
                            min_pos = cur_dist
                            min_nei = nei
                    lb1 = min_nei

                elif mode == 'min_value':
                    min_value = np.infty
                    min_nei = 0
                    for nei in neis:
                        cur_dist = (self.superpixels[nei].value - sp.value) ** 2
                        if cur_dist < min_value:
                            min_value = cur_dist
                            min_nei = nei
                    lb1 = min_nei

                self.merge(lb1, label, lb1, track_neighbors=track_neighbors)
                changed[label] = lb1
        return changed
