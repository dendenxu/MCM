import matplotlib.pyplot as plt
# import matplotlib as mpl
import numpy as np
from cell import SandCastleSimulator2D

# This method is deprecated due to large amount of computational resources required
# And this method requires a O(delta^width) time and space complexity, which is clearly unacceptable
# in typical computer science. However, beginning from this method, we can derive other simplified version
# that can compute a reasonable result(large resolution) in a reasonable amount of time(at least
# not exponential)

# Possible refinement includes:
# 1. Beginning from low resolution which can still be computed in time, we
# restrict the possible permutation by the value extracted from low resolution result
# from 10*10 to 1000*1000
# 2. Still beginning from low resolution permutation, but this time we won't run the simulation on low
# resolution values, instead we construct a cubic spline interpolation according to the value we've
# got from the permutation. Sand amount difference can be resolved by moving the whole edge of sand
# up or down. This method requires small computation resource since we've leveled down time complexity
# however, you gain and you lose. It also requires us to make the assumption that all sand in consideration
# are continuous and differentiable.

delta = 2
width = 10
depth = 10
pm = []
desired_sum = width * depth / 2
desired_loss_rate = 0.2
desired_loss = desired_sum * desired_loss_rate

#
# def next_level(l):
#     # print(len(pm))
#     if len(l) == width:
#         if sum(l) == desired_sum:
#             pm.append(l)
#         return
#     b = l[-1]
#     # left = b - delta
#     # right = b + delta + 1
#     left = b
#     right = b + 1
#     if len(l) > delta * 2:
#         d = np.asarray([l[i + 1] - l[i] for i in range(len(l) - delta * 2 - 1, len(l) - 1)])
#         if sum(d > 0) or sum(d < 0):
#             right = b + delta + 1
#             left = b - delta
#         elif d[-1] > 0:
#             right = b + delta + 1
#         else:
#             left = b - delta
#     elif len(l) > 1:
#         d = l[-1] - l[-2]
#         if d > 0:
#             right = b + delta + 1
#         else:
#             left = b - delta
#     left = left if left >= 0 else 0
#     right = right if right <= depth else depth
#     for i in range(left, right):
#         next_level(l + [i])
#
#
# for a in range(depth):
#     next_level([a, ])
#
# pm = np.asarray(pm)
# pm = pm[0:pm.shape[0] // 2 + 1, :]
#
# np.save("permutation_10x10_hermite.npy", pm)
#

# import matplotlib.pyplot as plt
# import numpy as np
# import imageio
# from scipy import interpolate
#
# pm = np.load("permutation_10x10_quarter.npy")
#
#
# def plot_for_offset(i):
#     print(i)
#     fig, ax = plt.subplots(figsize=(5, 5))
#     ax.plot(np.arange(100), 10 * (interpolate.CubicSpline(np.linspace(0, 10, 10), pm[i]))(np.linspace(0, 10, 100)))
#     ax.grid()
#
#     # IMPORTANT ANIMATION CODE HERE
#     # Used to keep the limits constant
#     ax.set_ylim((0, 100))
#
#     # Used to return the plot as an image rray
#     fig.canvas.draw()  # draw the canvas, cache the renderer
#     image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
#     image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#
#     return image
#
#
# kwargs_write = {'fps': 60.0, 'quantizer': 'nq'}
# imageio.mimsave('./powers_quarter.gif', [plot_for_offset(i) for i in range(pm.shape[0])], fps=60)
#

from scipy import interpolate

width *= 10
depth *= 10
desired_sum *= 100
desired_loss *= 100
pm = np.load("permutation_10x10.npy")
batch_size = delta * 2
pminter = []
[(pminter.append(10 * (interpolate.Akima1DInterpolator(np.linspace(0, 10, 10), pm[i]))(np.linspace(0, 10, 100)))) for i in
 range(pm.shape[0])]
pminter = pminter[::2]
pminter = pminter[::2]
pminter = pminter[::2]
counter = np.zeros(shape=(len(pminter),))
print(len(counter))
for index, edge in enumerate(pminter):
    sim = SandCastleSimulator2D(width, depth, delta * 2, edge)
    count = 0
    while True:
        for _ in range(batch_size):
            count += 1
            sim.wave()
        sim.update_life()
        sim.drop_dead()
        # print("One iteration done, enough?")
        if np.sum(sim.life) <= desired_sum - desired_loss:
            print("Enough! Enough for pm {} with counter value {}".format(index, count))
            counter[index] = count
            break
        # print("Not enough for pm {} at counter value {}".format(index, count))

# print(counter)
plt.figure()
plt.plot(counter)
plt.figure()
# plt.plot(pminter[np.argpartition(counter, -delta)[-delta]])
good = []
for i in range(1, 20):
    plt.plot(pminter[np.argpartition(counter, -i)[-i]])
    # good.append(pminter[np.argpartition(counter, -i)[-i]])
good = np.asarray(good)
plt.ylim((0, 100))

