import matplotlib.pyplot as plt
import numpy as np
from cell import SandCastleSimulator2D

delta = 2
width = 100
depth = 50
desired_sum = np.pi * (width / 4) ** 2 / 2
desired_loss_rate = 0.4
desired_loss = desired_sum * desired_loss_rate
batch_size = 2

circle = np.zeros((0, width))

for i in np.linspace(0.5, 1.5, 200):
    last = np.sqrt(1 - np.linspace(-1, 1, int((width / 2) / i)) ** 2) * (width / 4) * i
    rest = width - last.shape[0]
    left = rest // 2
    right = rest - left
    last = np.insert(last, -1, values=np.zeros((right,)))
    last = np.insert(last, 0, values=np.zeros((left,)))
    circle = np.append(circle, [last], axis=0)

plt.figure(figsize=(10, 5))
plt.ylim((0, depth))
for i, _ in enumerate(circle):
    plt.plot(circle[i])

counter = np.ndarray((circle.shape[0],))
water = np.ndarray((circle.shape[0],))

# for index, edge in enumerate(circle):
#     sim = SandCastleSimulator2D(
#         width, depth, delta * 2, edge, shear_rate=1, humidify_rate=0.005, humidify_depth=2, initial_humidity=0.5, slow=False)
#     count = 0
#     while True:
#         for _ in range(batch_size):
#             count += 1
#             sim.wave()
#         sim.update_life()
#         sim.drop_dead()
#         # print("One iteration done, enough?")
#         if np.sum(sim.life) <= desired_sum - desired_loss:
#             print("Enough! Enough for pm {} with counter value {}".format(index, count))
#             counter[index] = count
#             break
#         # print("Not enough for pm {} at counter value {}".format(index, count))


batch_size = 20
for index, edge in enumerate(circle):
    sim = SandCastleSimulator2D(
        width, depth, delta * 2, edge, shear_rate=1, humidify_rate=0.05, humidify_depth=2, initial_humidity=0.5, slow=False)
    for _ in range(batch_size):
        sim.wave()
    sim.update_life()
    sim.drop_dead()
    counter[index] = np.sum(sim.life)
    water[index] = (np.sum(sim.humidity) + counter[index] - width * depth) / counter[index]
plt.figure(figsize=(10, 10))
plt.plot(counter)
plt.figure(figsize=(10, 10))
plt.plot(water)
