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
