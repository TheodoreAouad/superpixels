import matplotlib.pyplot as plt
import src.plotter as p
fig, ax = plt.subplots(1, 1, figsize=(15, 15))

plt.imsave(f'./results/hiersup/first_try/{len(self.superpixels)}/sp.png', self.superpixels.array_means, cmap='gray')
plt.imsave(f'./results/hiersup/first_try/{len(self.superpixels)}/img.png', self.superpixels.img_ini, cmap='gray')

p.plot_img_mask_on_ax(ax, self.superpixels.array_means, self.superpixels.infer_superpixel_edges())
fig.savefig(f'./results/hiersup/first_try/{len(self.superpixels)}/sp_frontier.png')

p.plot_img_mask_on_ax(ax, self.superpixels.img_ini, self.superpixels.infer_superpixel_edges())
fig.savefig(f'./results/hiersup/first_try/{len(self.superpixels)}/img_frontier.png')


import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
ax.scatter(self.nb_neighbors, self.time_update)
ax.set_yscale('log')
ax.set_xscale('log')
fig.savefig('./results/hiersup/first_try/nei_time.png')

fig, ax = plt.subplots(1, 1, figsize=(15, 15))
ax.plot(self.time_weights)
ax.set_yscale('log')
fig.savefig('./results/hiersup/first_try/time_weights.png')

fig, ax = plt.subplots(1, 1, figsize=(15, 15))
ax.plot(self.time_update)
ax.set_yscale('log')
fig.savefig('./results/hiersup/first_try/time_update.png')

fig, ax = plt.subplots(1, 1, figsize=(15, 15))
ax.plot(self.time_merge)
ax.set_yscale('log')
fig.savefig('./results/hiersup/first_try/time_merge.png')

fig, ax = plt.subplots(1, 1, figsize=(15, 15))
ax.plot(self.time_step)
ax.set_yscale('log')
fig.savefig('./results/hiersup/first_try/time_step.png')


fig, ax = plt.subplots(1, 1, figsize=(15, 15))
ax.plot(self.time_tot, label='tot')
ax.plot(np.cumsum(self.time_weights), label='weights')
ax.plot(np.cumsum(self.time_update), label='update')
ax.plot(np.cumsum(self.time_merge), label='merge')
ax.legend()
fig.savefig('./results/hiersup/first_try/time_tot.png')


fig, ax = plt.subplots(1, 1, figsize=(15, 15))
ax.plot(self.nb_neighbors)
ax.set_yscale('log')
fig.savefig('./results/hiersup/first_try/nb_neighbors.png')

