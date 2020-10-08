import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

from src.plotter import plot_img_mask_on_ax


class SLIC:

    def __init__(self, K: int, m: float, thresh: float, maxit: int):

        self.centers = None
        self.next_centers = None
        self.N = None
        self.W = None
        self.L = None
        self.pixel_labels = None
        self.pixel_distance = None
        self.clusters = None


        self.K = K
        self.m = m
        self.thresh = thresh
        self.maxit = maxit


    def fit(self, img: np.ndarray, show_progress: bool = False):
        self.W, self.L = img.shape
        self.N = self.W * self.L
        self.pixel_labels = np.zeros_like(img).astype(int)
        self.pixel_distance = np.zeros_like(img) + np.infty

        self.centers = self._init_centers(img, self.S)

        error = self.thresh + 1
        it = 0
        while error > self.thresh and it < self.maxit:

            for center_label, center in enumerate(self.centers):
                _, center_i, center_j = center.astype(int)


                for i in range(max(0, center_i-self.S), min(self.W, center_i + self.S)):
                    for j in range(max(0, center_j-self.S), min(self.L, center_j + self.S)):
                        cur_dist = self.compute_distance(center, [img[i, j], i, j])

                        if cur_dist < self.pixel_distance[i, j]:
                            self.pixel_distance[i, j] = cur_dist
                            self.pixel_labels[i, j] = center_label

            self.next_centers = self._update_centers(img)
            error = self.compute_error(self.centers, self.next_centers)
            self.centers = self.next_centers

            if show_progress:
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                plot_img_mask_on_ax(axs[0], img, self.infer_superpixel_edges())
                axs[1].imshow(self.infer_superpixel_img(), cmap="gray")

            print(f"Current iteration: {it+1} / {self.maxit}. Error: {error}")
            it += 1


    @staticmethod
    def _init_centers(img: np.ndarray, S: float):
        centers = []
        W, L = img.shape

        i = 0
        while i < W:
            j = 0
            while j < L:
                centers.append(np.array([
                    img[i, j], i, j
                ]))

                j += S
            i += S

        return np.stack(centers)

    def _update_centers(self, img):
        next_centers = np.zeros((len(self.centers), 3))
        nb_pixels_centers = np.zeros((len(self.centers), 1))
        for i in range(self.W):
            for j in range(self.L):
                label = self.pixel_labels[i, j]
                next_centers[label] += np.array([
                    img[i, j], i, j
                ])
                nb_pixels_centers[label, 0] += 1

        return next_centers / nb_pixels_centers


    def infer_superpixel_img(self):
        mask = self.pixel_labels + 0
        for label in range(len(self.centers)):
            mask[mask == label] = self.centers[label][0]
        return mask


    def infer_superpixel_edges(self):
        mask = self.pixel_labels + 0

        ker1 = np.array([[0, 1, -1]])
        ker2 = np.array([[0], [1], [-1]])

        edge1 = ndimage.convolve(mask, ker1) != 0
        edge2 = ndimage.convolve(mask, ker2) != 0

        edge1[edge2] = 1

        return edge1


    @staticmethod
    def compute_error(centers: np.ndarray, next_centers: np.ndarray):
        error = 0
        for center1, center2 in zip(centers, next_centers):
            error += np.linalg.norm(center1 - center2)
        return error

    def compute_distance(self, p1, p2):
        dc2 = (p1[0] - p2[0])**2
        ds = np.linalg.norm(p1[1:] - p2[1:])
        return np.sqrt(dc2 + (ds * self.m / self.S) ** 2)

    @property
    def S(self):
        return int(np.sqrt(self.N / self.K))
