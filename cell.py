import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


class SandCastleSimulator2D:
    # Initialize the sand castle randomly
    def __init__(self, width, depth, delta):
        self.delta = delta
        assert (width > (self.delta * 2))
        self.width = width
        self.depth = depth
        # The front of our sand castle
        # self.edge = np.random.normal(width / 2, width / 4, depth).round(0)
        # self.edge = depth - ((np.cos(np.linspace(-np.pi, np.pi, width)) + 1) * depth / 2).round(0)
        self.edge = depth - ((np.cos(np.linspace(-np.pi * 3, np.pi * 3, width)) + 1) * depth / 10 + depth / 2).round(0)
        # self.edge = depth - (np.cos(np.linspace(0, np.pi / 2, width)) * depth).round(0)
        # self.edge = depth - ((np.sqrt(1 - np.linspace(-1, 1, width) ** 2)) * depth / 2).round(0)
        self.edge = np.asarray([0 if x < 0 else (depth - 1 if x > depth - 1 else x) for x in self.edge], dtype=int)
        # self.edge.sort()
        self.life = np.array([[i >= self.edge[j] for j in range(width)] for i in range(depth)])
        self.humidity = np.full((depth, width), fill_value=0.1)
        # self.humidity[0:int(width / 2), :] = np.full((int(width / 2), depth), fill_value=0.5)
        self.slope = np.zeros((width,), float)
        self.update_slope()
        sea_blue = np.array([0, 102, 153, 256]) / 256
        sandy_brown = np.array([244, 164, 96, 256]) / 256
        self.sea_sand = mpl.colors.ListedColormap(np.concatenate((np.tile(sea_blue, 128).reshape(128, 4),
                                                                  np.tile(sandy_brown, 128).reshape(128, 4)), axis=0))
        hot = plt.cm.get_cmap('Oranges', 256)
        sea_wet_sand = hot(np.linspace(0, 1, 256))
        sea_wet_sand[0:128] = sea_wet_sand[128::]
        sea_wet_sand[128::] = np.tile(sea_blue, 128).reshape(128, 4)
        self.sea_wet_sand = mpl.colors.ListedColormap(sea_wet_sand)

        self.shear_rate = 0.5
        self.humidify_rate = 0.005
        self.humidify_depth = 1
        self.osmotic_rate = 0.005
        self.osmotic_depth = 10
        self.longest = np.sqrt(2) * self.osmotic_depth * 2
        self.osmosis()
        self.drop_dead()

        self.delta_humidity = np.flip(
            np.exp(np.linspace(-10 * self.humidify_depth, 0, self.depth))) * self.humidify_rate
        self.delta_humidity = np.tile(self.delta_humidity, self.width).reshape(self.width, self.depth)
        self.delta_humidity = np.transpose(self.delta_humidity)

    def osmosis(self):
        osmotic_depth = self.osmotic_depth
        longest = self.longest
        # longest = np.sqrt(self.width ** 2 + self.depth ** 2)
        for i in range(osmotic_depth, self.width - osmotic_depth, self.delta):
            for j in range(self.edge[i] - osmotic_depth,
                           self.edge[i] + osmotic_depth):
                for k in range(i - osmotic_depth,
                               i + osmotic_depth):
                    if j < 0 or j >= self.depth:
                        continue
                    distance = np.sqrt((j - self.edge[i]) ** 2 + (k - i) ** 2)
                    delta_osmosis = np.exp(-10 * distance / longest) * self.osmotic_rate
                    # print("delta_osmosis is {}".format(delta_osmosis))
                    self.humidity[j, k] += delta_osmosis

    def update_slope(self):
        slope = np.asarray([(self.edge[i + self.delta] - self.edge[i - self.delta]) / (2 * self.delta)
                            for i in range(self.delta, self.width - self.delta)])
        # for d in range(1, self.delta):
        #     slope[d: self.width - self.delta * 2 - d] = \
        #     np.asarray([((self.edge[i + d] - self.edge[i - d]) / (2 * d) + slope[i]) / 2
        #                 for i in range(d, self.width - self.delta * 2 - d)])
        slope = np.insert(slope, 0, np.full([self.delta, ], slope[0]))
        slope = np.append(slope, np.full([self.delta, ], slope[-1]))
        self.slope = np.arctan(abs(slope))

    def update_life(self):
        self.life = np.array([[i >= self.edge[j] for j in range(self.width)] for i in range(self.depth)])

    def drop_dead(self):
        self.humidity = np.array(
            [[self.humidity[i, j] if self.life[i, j] else 1.5 for j in range(self.width)] for i in range(self.depth)])

    def drop_prick(self):
        delta = self.delta * 2
        edge = self.edge
        for i in range(delta, self.width - delta):
            neighbors = np.concatenate([self.edge[i - delta:i], self.edge[i + 1:i + delta + 1]])
            # print("Current edge[i] is {}".format(edge[i]))
            # print("Current neighbors are {}".format(neighbors))
            if edge[i] < min(neighbors):
                edge[i] = max(neighbors)
        self.edge = edge
        # self.life = np.array([[i >= self.edge[j] for j in range(self.width)] for i in range(self.depth)])

    def wave(self):
        # Let's assume the slope and humidity level and the angle of the
        # impact will change what our sand castle look like

        cosine = abs(np.cos(self.slope))
        sine = abs(np.sin(self.slope))

        # Then, according to the humidity level and angle of impact,
        # we'll determine the san cell that is currently alive
        delta_shear = sine * self.shear_rate
        for k in range(self.delta):
            for i in range(self.width):
                # print("Current value of i is {}".format(i))
                # print("Current value of edge is {}".format(self.edge[i]))
                if (delta_shear[i] + 1 / (1 - self.humidity[self.edge[i], i]) - 1) > 1:
                    # self.life[self.edge[i], i] = False
                    self.edge[i] += 1 if self.edge[i] < self.depth - 1 else 0
                    delta_shear[i] -= delta_shear[i] / self.delta
            self.drop_prick()

        # We'll firstly update the humidity level according to the angle of impact
        delta_humidity = self.delta_humidity * cosine
        # print(delta_humidity.shape)
        # print(self.humidity.shape)
        for i in range(self.width):
            # print("Current i value is {}, and current edge value is {}".format(i, self.edge[i]))
            # print("Humidity from {} to {} is changed by {} to {} of delta humidity"
            #       .format(self.edge[i], self.depth, 0, self.depth - self.edge[i] - 1))
            self.humidity[self.edge[i]::, i] += delta_humidity[0:self.depth - self.edge[i], i]
        self.osmosis()
        # Finally, we'll update the sand castle's humidity information and slope information
        self.update_slope()
        # self.drop_dead()


# Please run them in interactive mode to get proper image output
# ipython is recommended
# from cell import *
# delta should be no smaller than 1
sim = SandCastleSimulator2D(1000, 1000, 3)
fig = plt.figure(figsize=(16, 12))
fig.add_subplot(231)
plt.imshow(sim.humidity, cmap=sim.sea_wet_sand)
# plt.figure()
# plt.imshow(sim.life, cmap=sim.sea_sand)
# sim.wave()

for i in range(5):
    for _ in range(100):
        sim.wave()
    sim.update_life()
    sim.drop_dead()
    fig.add_subplot(230 + i + 2)
    plt.imshow(sim.humidity, cmap=sim.sea_wet_sand)

plt.savefig("cell.eps")

# plt.figure()
# plt.imshow(sim.life, cmap=sim.sea_sand)
