import numpy as np


def plot_img_mask_on_ax(ax, img, mask, alpha=.7):

    masked = np.ma.masked_where(mask == 0, mask)
    ax.imshow(img, cmap='gray')
    ax.imshow(masked, cmap='jet', alpha=alpha)

    return ax
