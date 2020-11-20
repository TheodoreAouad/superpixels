import matplotlib.pyplot as plt
import src.plotter as p
fig, ax = plt.subplots(1, 1, figsize=(15, 15))

plt.imsave(f'./results/hiersup/first_try/{len(self.superpixels)}/sp.png', self.superpixels.array_means, cmap='gray')
plt.imsave(f'./results/hiersup/first_try/{len(self.superpixels)}/img.png', self.superpixels.img_ini, cmap='gray')

p.plot_img_mask_on_ax(ax, self.superpixels.array_means, self.superpixels.infer_superpixel_edges())
fig.savefig(f'./results/hiersup/first_try/{len(self.superpixels)}/sp_frontier.png')

p.plot_img_mask_on_ax(ax, self.superpixels.img_ini, self.superpixels.infer_superpixel_edges())
fig.savefig(f'./results/hiersup/first_try/{len(self.superpixels)}/img_frontier.png')