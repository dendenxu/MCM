import matplotlib.pyplot as plt
import matplotlib as mpl
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
delta = 3
width = 100
depth = 100
permutation = []
desired_sum = width * depth / 2
desired_loss_rate = 0.5
desired_loss = desired_sum * desired_loss_rate


def next_level(l):
    print(len(permutation))
    if len(l) == width:
        if sum(l) == desired_sum:
            permutation.append(l)
        return
    b = l[-1]
    left = b
    right = b + 1
    if len(l) > delta * 2:
        d = np.asarray([l[i + 1] - l[i] for i in range(len(l) - delta * 2 - 1, len(l) - 1)])
        if sum(d > 0) or sum(d < 0):
            right = b + delta + 1
            left = b - delta
        elif d[-1] > 0:
            right = b + delta + 1
        else:
            left = b - delta
    elif len(l) > 1:
        d = l[-1] - l[-2]
        if d > 0:
            right = b + delta + 1
        else:
            left = b - delta
    left = left if left >= 0 else 0
    right = right if right <= depth else depth
    for i in range(left, right):
        next_level(l + [i])


for a in range(depth):
    next_level([a, ])

permutation = np.asarray(permutation)
permutation = permutation[0:permutation.shape[0] // 2 + 1, :]
counter = np.zeros(shape=(permutation.shape[0],))

batch_size = delta
for index, edge in enumerate(permutation):
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
            # print("Enough! Enough for permutation {} with counter value {}".format(edge, count))
            counter[index] = count
            break
        # print("Not enough for permutation {} at counter value {}".format(edge, count))

# print(counter)
plt.figure()
plt.plot(counter)
plt.figure()
plt.plot(permutation[np.argpartition(counter, -delta)[-delta]])
