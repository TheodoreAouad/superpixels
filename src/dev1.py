import pathlib
import matplotlib.pyplot as plt
import src.plotter as p
fig, ax = plt.subplots(1, 1, figsize=(15, 15))

pathlib.Path(f'./results/hiersup/try_13/{len(self.superpixels)}/').mkdir(exist_ok=True, parents=True)

plt.imsave(f'./results/hiersup/try_13/{len(self.superpixels)}/sp.png', self.superpixels.array_means, cmap='gray')
plt.imsave(f'./results/hiersup/try_13/{len(self.superpixels)}/img_ini.png', self.superpixels.img_ini, cmap='gray')

p.plot_img_mask_on_ax(ax, self.superpixels.array_means, self.superpixels.infer_superpixel_edges())
fig.savefig(f'./results/hiersup/try_13/{len(self.superpixels)}/sp_frontier.png')

p.plot_img_mask_on_ax(ax, self.superpixels.img_ini, self.superpixels.infer_superpixel_edges())
fig.savefig(f'./results/hiersup/try_13/{len(self.superpixels)}/img_frontier.png')

import pathlib
import matplotlib.pyplot as plt
import src.plotter as p
# dist, (lb1, lb2) = self.weights[0]
lb1, lb2 = (label1, label2)
pathlib.Path(f'./results/hiersup/try_13/{len(self.superpixels)}/').mkdir(exist_ok=True, parents=True)
plt.imsave(f'./results/hiersup/try_13/{len(self.superpixels)}/some_sp.png', self.superpixels.array_some_sp([lb1, lb2]))

import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(self.nb_neighbors, self.time_update)
# ax.set_yscale('log')
# ax.set_xscale('log')
ax.set_xlabel('Nb neighbors')
ax.set_ylabel('Time Update')
ax.grid(True)
fig.savefig('./results/hiersup/try_13/nei_time.png')



fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(self.time_weights)
ax.set_yscale('log')
fig.savefig('./results/hiersup/try_13/time_weights.png')

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(self.time_update)
ax.set_yscale('log')
fig.savefig('./results/hiersup/try_13/time_update.png')


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(self.time_merge)
ax.set_yscale('log')
fig.savefig('./results/hiersup/try_13/time_merge.png')

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(self.time_step)
ax.set_yscale('log')
fig.savefig('./results/hiersup/try_13/time_step.png')


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(self.time_tot, label='tot')
ax.plot(np.cumsum(self.time_weights), label='weights')
ax.plot(np.cumsum(self.time_update), label='update')
ax.plot(np.cumsum(self.time_merge), label='merge')
ax.plot(np.cumsum(self.time_update) + np.cumsum(self.time_weights) + np.cumsum(self.time_merge), label='sum')
ax.legend()
ax.grid(True)
fig.savefig('./results/hiersup/try_13/time_tot.png')


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(self.nb_neighbors)
fig.savefig('./results/hiersup/try_13/nb_neighbors.png')



fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(self.nb_weights_pop, self.time_weights)
# ax.set_yscale('log')
# ax.set_xscale('log')
ax.set_xlabel('Nb weights pop')
ax.set_ylabel('Weights finding time')
fig.savefig('./results/hiersup/try_13/weights_pop_time.png')


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(self.nb_weights_pop)
ax.set_yscale('log')
ax.set_xscale('log')
ax.grid(True)
fig.savefig('./results/hiersup/try_13/nb_weights_pop.png')


from time import time

t_1 = time()
for _ in range(10000000):
    keep = (label1 in self.merged_away) or (label2 in self.merged_away)
du = time() - t_1
du

t_1 = time()
for _ in range(10000000):
    keep = (label1 is None) or (label2 is None)
du2 = time() - t_1


daz = set()
t_1 = time()
for _ in range(10000000):
    daz.add(_)
du3 = time() - t_1


daz = 0
t_1 = time()
for _ in range(10000000):
    daz = None
du4 = time() - t_1


for idx, (dist, key) in enumerate(self.weights):
    if key == (256, 512):
        print(idx)
        break
print('Done')


done = set()
for dist, key in self.weights:
    if key not in done:
        done.add(key)

done2 = set()
for dist, key in weights2:
    if key not in done2:
        done2.add(key)

t1 = time()
weights1, _, _ = self.compute_weights_from_img(self.superpixels, self.distance)
t2 = time()
weights2, _, _ = self.compute_weights(self.superpixels, self.distance)
t3 = time()

d1 = t2 - t1
d2 = t3 - t2

diff = []
for w1, w2 in zip(weights1, weights2):
    if w1 != w2:
        diff.append((w1, w2))

for idx, w1 in enumerate(weights1):
    weights1[idx] = (w1[0], tuple(sorted(w1[1])))

diff2 = []
for w1, w2 in diff:
    if (w1 not in weights2) or (w2 not in weights1):
        diff2.append((w1, w2))

sizes = []
for sp in self.superpixels.superpixels:
    sizes.append(len(self.superpixels[sp]))

plt.imsave(f'./results/hiersup/try_13/{len(self.superpixels)}/some_sp.png', self.superpixels.array_some_sp([3770]))
# plt.imsave(f'./results/hiersup/try_13/{len(self.superpixels)}/labels.png', self.superpixels.array_label)
