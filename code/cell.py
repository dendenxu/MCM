import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


class SandCastleSimulator2D:
    # Initialize the sand castle randomly
    def __init__(self, width, depth, delta,
                 initial_edge=None, shear_rate=1,
                 humidify_rate=0.005, humidify_depth=1, initial_humidity=0.1,
                 slow=True, osmosis=False):
        self.delta = delta
        assert (width > (self.delta * 2))
        self.width = width
        self.depth = depth
        # The front of our sand castle
        if initial_edge is None:
            self.edge = depth - (
                    (np.cos(np.linspace(-np.pi * 3, np.pi * 3, width)) + 1) * depth / 10 + depth / 2).round(0)
            # self.edge = np.random.normal(width / 2, width / 4, depth).round(0)
            # self.edge = depth - ((np.cos(np.linspace(-np.pi, np.pi, width)) + 1) * depth / 2).round(0)
            # self.edge = depth - (np.cos(np.linspace(0, np.pi / 2, width)) * depth).round(0)
            # self.edge = depth - ((np.sqrt(1 - np.linspace(-1, 1, width) ** 2)) * depth / 2).round(0)
        else:
            self.edge = initial_edge
        self.edge = np.asarray([0 if x < 0 else (depth - 1 if x > depth - 1 else x) for x in self.edge], dtype=int)
        self.slow = slow
        self.life = np.array([[i >= self.edge[j] for j in range(width)] for i in range(depth)])
        self.humidity = np.full((depth, width), fill_value=initial_humidity)
        self.slope = np.zeros((width,), float)
        self.update_slope()
        sea_blue = np.array([0, 102, 153, 256]) / 256
        sandy_brown = np.array([244, 164, 96, 256]) / 256
        self.sea_sand = mpl.colors.ListedColormap(
            np.concatenate((np.tile(sea_blue, 128).reshape(128, 4),
                            np.tile(sandy_brown, 128).reshape(128, 4)), axis=0))
        hot = plt.cm.get_cmap('Oranges', 256)
        sea_wet_sand = hot(np.linspace(0, 1, 256))
        sea_wet_sand[0:128] = sea_wet_sand[128::]
        sea_wet_sand[128::] = np.tile(sea_blue, 128).reshape(128, 4)
        self.sea_wet_sand = mpl.colors.ListedColormap(sea_wet_sand)

        self.shear_rate = shear_rate
        self.humidify_rate = humidify_rate
        self.humidify_depth = humidify_depth
        self.osmotic_rate = 0.005
        self.osmotic_depth = 5
        self.longest = np.sqrt(2) * self.osmotic_depth * 2
        self.osmo = osmosis

        if self.osmo:
            self.osmosis()
        self.drop_dead()

        self.delta_humidity = np.flip(
            np.exp(np.linspace(-10 / self.humidify_depth, 0, self.depth))) * self.humidify_rate
        self.delta_humidity = np.tile(self.delta_humidity, self.width).reshape(self.width, self.depth)
        self.delta_humidity = np.transpose(self.delta_humidity)

    def osmosis(self):
        osmotic_depth = self.osmotic_depth
        longest = self.longest
        for i in range(osmotic_depth, self.width - osmotic_depth, self.delta):
            for j in range(self.edge[i] - osmotic_depth,
                           self.edge[i] + osmotic_depth):
                for k in range(i - osmotic_depth,
                               i + osmotic_depth):
                    if j < 0 or j >= self.depth:
                        continue
                    distance = np.sqrt((j - self.edge[i]) ** 2 + (k - i) ** 2)
                    delta_osmosis = np.exp(-10 * distance / longest) * self.osmotic_rate
                    self.humidity[j, k] += delta_osmosis

    def update_slope(self):
        slope = np.asarray([(self.edge[i + self.delta] - self.edge[i - self.delta]) / (2 * self.delta)
                            for i in range(self.delta, self.width - self.delta)])
        slope = np.insert(slope, 0, np.full([self.delta, ], slope[0]))
        slope = np.append(slope, np.full([self.delta, ], slope[-1]))
        self.slope = np.arctan(abs(slope))

    def update_life(self):
        self.life = np.array([[i >= self.edge[j] for j in range(self.width)] for i in range(self.depth)])

    def drop_dead(self):
        self.humidity = np.array(
            [[self.humidity[i, j] if self.life[i, j] else 1.5
              for j in range(self.width)] for i in range(self.depth)])

    def drop_prick(self):
        delta = self.delta * 2
        edge = self.edge
        for i in range(delta, self.width - delta):
            neighbors = np.concatenate([self.edge[i - delta:i], self.edge[i + 1:i + delta + 1]])
            if edge[i] < min(neighbors):
                edge[i] = max(neighbors)
        self.edge = edge

    def wave(self):

        cosine = abs(np.cos(self.slope))
        sine = abs(np.sin(self.slope))

        delta_shear = sine * self.shear_rate
        for k in range(self.delta * (1 if self.slow else 2)):
            for i in range(self.width):
                if ((cosine[i] > np.sqrt(2) / 2 or k < 1) or self.slow) and (
                        delta_shear[i] + 1 / (1 - self.humidity[self.edge[i], i]) - 1) > 1:
                    self.edge[i] += 1 if self.edge[i] < self.depth - 1 else 0
                    delta_shear[i] -= delta_shear[i] / self.delta
            self.drop_prick()
        if self.osmo:
            self.osmosis()
        delta_humidity = self.delta_humidity * cosine
        for i in range(self.width):
            self.humidity[self.edge[i]::, i] += delta_humidity[0:self.depth - self.edge[i], i]
        self.update_slope()

# sim = SandCastleSimulator2D(1000, 1000, 3, humidify_depth=1, shear_rate=0.75)
# fig = plt.figure(figsize=(20, 20))
# fig.add_subplot(231)
# plt.imshow(sim.humidity, cmap=sim.sea_wet_sand)


# for i in range(5):
#     for _ in range(100):
#         sim.wave()
#     sim.update_life()
#     sim.drop_dead()
#     fig.add_subplot(230 + i + 2)
#     plt.imshow(sim.humidity, cmap=sim.sea_wet_sand)

# plt.savefig("cell.eps")
