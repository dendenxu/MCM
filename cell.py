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
        self.edge = width - (np.cos(np.linspace(0, np.pi / 2, width)) * depth).round(0)
        self.edge = [0 if x < 0 else (width - 1 if x > width - 1 else x) for x in self.edge]
        # self.edge.sort()
        self.life = np.array([[i >= self.edge[j] for j in range(width)] for i in range(depth)])
        self.humidity = np.full((width, depth), fill_value=0.0)
        self.humidity[0:int(width / 2), :] = np.full((int(width / 2), depth), fill_value=0.5)
        self.drop_dead()
        self.slope = self.calculate_slope()
        sea_blue = np.array([0, 102, 153, 256]) / 256
        sandy_brown = np.array([244, 164, 96, 256]) / 256
        self.sea_sand = mpl.colors.ListedColormap(np.concatenate((np.tile(sea_blue, 128).reshape(128, 4),
                                                                  np.tile(sandy_brown, 128).reshape(128, 4)), axis=0))
        hot = plt.cm.get_cmap('Oranges', 256)
        sea_wet_sand = hot(np.linspace(0, 1, 256))
        sea_wet_sand[0:160] = sea_wet_sand[256-160::]
        sea_wet_sand[160::] = np.tile(sea_blue, 256-160).reshape(256-160, 4)
        self.sea_wet_sand = mpl.colors.ListedColormap(sea_wet_sand)

    def calculate_slope(self):
        slope = np.asarray([(self.edge[i + self.delta] - self.edge[i - self.delta]) / (2 * self.delta)
                            for i in range(self.delta, len(self.edge) - self.delta)])
        slope = np.insert(slope, 0, np.full([self.delta, ], slope[0]))
        slope = np.append(slope, np.full([self.delta, ], slope[-1]))
        return np.arctan(slope)

    def drop_dead(self):
        self.humidity = np.array(
            [[self.humidity[i, j] if self.life[i, j] else 1. for j in range(self.width)] for i in range(self.depth)])

    def wave(self):
        pass


# Please run them in interactive mode to get proper image output
# ipython is recommended
# from cell import *
# sim = SandCastleSimulator2D(1000, 1000, 5)
# plt.figure()
# plt.imshow(sim.humidity, cmap=sim.sea_wet_sand)
# plt.figure()
# plt.imshow(sim.life, cmap=sim.sea_sand)
