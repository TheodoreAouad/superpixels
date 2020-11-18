from __future__ import annotations
from typing import List, Optional

from scipy.ndimage import convolve
import numpy as np


class Superpixel:

    def __init__(
        self,
        label: Optional[int] = None,
        pixel_idxs: Optional[List] = None,
        values: Optional[List] = None,
    ):
        self.label = label
        self.pixel_idxs = pixel_idxs
        self.values = values

    def value(self):
        return np.mean(self.values)

    def __getitem__(self, idx):
        return self.pixel_idxs[idx], self.values[idx]

    def merge(self, sp: Superpixel, label: Optional[int] = None) -> Superpixel:
        return Superpixel(
            label=self.label if label is None else label,
            pixel_idxs=self.pixel_idxs + sp.pixel_idxs,
            values=self.values + sp.values,
        )

    def __repr__(self):
        return (
            f"Superpixel [\n"
            f"   Label: {self.label}\n"
            f"   Pixels: {self.pixel_idxs}\n"
            f"   Values: {self.values}\n"
            "]"
        )


class SuperpixelImage:

    def __init__(
        self,
        img: np.ndarray,
        superpixels: Optional[List[Superpixel]] = None,
        mask: Optional[np.ndarray] = None,
    ):
        self.img_ini = img + 0

        if superpixels is not None:
            self.superpixels = superpixels
        elif mask is not None:
            self.superpixels = self.init_sp_from_mask(img, mask)
        else:
            self.superpixels = self.init_sp_from_img(img)

        self.array_label = self.compute_array_label()
        self.array_means = self.compute_array_means()

    def __getitem__(self, idx):
        return self.superpixels[idx]

    def __len__(self):
        return len(self.superpixels)

    def compute_array_label(self) -> np.ndarray:
        img = np.zeros(self.img_ini.shape)

        for sp in self.superpixels:
            for (idx1, idx2), _ in sp:
                img[idx1, idx2] = sp.label

        return img

    def compute_array_means(self) -> np.ndarray:
        img = np.zeros(self.img_ini.shape)

        for sp in self.superpixels:
            value = sp.value()
            for (idx1, idx2), pixel_id in sp:
                img[idx1, idx2] = value

        return img


    @staticmethod
    def init_sp_from_img(img: np.ndarray) -> List[Superpixel]:
        W, L = img.shape
        superpixels = []

        idx = 0
        for i in range(W):
            for j in range(L):
                superpixels.append(Superpixel(
                    label=idx,
                    pixel_idxs=[[i, j]],
                    values=[img[i, j]],
                ))
                idx += 1

        return superpixels

    @staticmethod
    def init_sp_from_mask(img: np.ndarray, mask: np.ndarray) -> List[Superpixel]:
        W, L = mask.shape
        superpixels = []

        idx = 0
        for value in np.unique(mask):
            superpixels.append(Superpixel(
                label=idx,
                pixel_idxs=list(zip(*np.where(mask==value))),
                values=list(img[np.where(mask==value)])
            ))
            idx += 1

        return superpixels


    def infer_superpixel_edges(self) -> np.ndarray:
        mask = self.array_label

        ker1 = np.array([[0, 1, -1]])
        ker2 = np.array([[0], [1], [-1]])

        edge1 = convolve(mask, ker1) != 0
        edge2 = convolve(mask, ker2) != 0

        edge1[edge2] = 1

        return edge1

    def __repr__(self):
        # res = "SuperpixelImage([\n"
        # for sp in self.superpixels:
        #     toadd = "   " + sp.__repr__().replace('\n', '\n   ')
        #     res += toadd + ",\n"
        # res += "])"
        # return res
        return super().__repr__().replace('>', f' ({len(self)} superpixels)>')
