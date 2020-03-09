import numpy as np
import matplotlib.pyplot as plt
from cell import SandCastleSimulator2D

delta = 2
width = 100
depth = 100
desired_loss_rate = 0.2
batch_size = 2
counter = np.ndarray((50, ))
batch_size = 2
for index, water in enumerate(np.linspace(0.2, 0.8, 50)):
    sim = SandCastleSimulator2D(width, depth, delta, shear_rate=0.75, initial_humidity=water, slow=False)
    count = 0
    desired_sum = np.sum(sim.life)
    while True:
        for _ in range(batch_size):
            count += 1
            sim.wave()
        sim.update_life()
        sim.drop_dead()
        if np.sum(sim.life) / desired_sum < 1 - desired_loss_rate:
            counter[index] = count
            print("Water-sand proportion {} is done.".format(water))
            break
import numpy as np
from scipy import interpolate
import matplotlib as mpl
mpl.rc('font', family='Garamond')
# inter = np.load("inter.npy")
# counter = np.load("counter.npy")
inter = (interpolate.CubicSpline(np.linspace(0, 50, 10), counter[::5]))(np.linspace(0, 50, 500))
plt.figure(figsize=(12, 6))
plt.plot(np.linspace(0.01, 0.2, 50), counter, label="water-sand proportion", linewidth=2.5, alpha=0.7, color="blue")
plt.plot(np.linspace(0.01, 0.2, 500), inter, label="interpolation", linewidth=2.5, alpha=0.7, color="red")
plt.legend()
#argmax = np.argmax(inter)/10
#max = np.max(inter)
#plt.scatter([argmax, ], [max, ], 100, color='red', alpha=0.7)
#plt.plot([argmax, argmax], [0, max], color='red', linewidth=1.5, linestyle="--", alpha=0.7)
#plt.plot([0, argmax], [max, max], color='red', linewidth=1.5, linestyle="--", alpha=0.7)
ax = plt.gca()
ax.spines["right"].set_color("none")
ax.spines["top"].set_color("none")

# plt.annotate(r'Most long lasting',
#              xy=(argmax, max), xycoords='data',
#              xytext=(50, -50), textcoords='offset points', fontsize=16,
#              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plt.xlabel("Water-sand Proportion Rate", fontsize=16)
plt.ylabel("Iterations Before Falling Apart", fontsize=16)
plt.legend(loc='upper right', prop={"size": "14"})
# plt.yticks(np.linspace(5, 25, 5, endpoint=True))
# plt.xticks(np.linspace(25, 200, 8, endpoint=True))
ax.xaxis.set_ticks_position("bottom")
ax.spines["bottom"].set_position(("data", 0))
ax.yaxis.set_ticks_position("left")
ax.spines["left"].set_position(("data", 0))
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(16)
    label.set_bbox(dict(facecolor='white', edgecolor="none", alpha=0.5))