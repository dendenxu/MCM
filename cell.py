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
        self.edge = np.asarray([0 if x < 0 else (width - 1 if x > width - 1 else x) for x in self.edge], dtype=int)
        # self.edge.sort()
        self.life = np.array([[i >= self.edge[j] for j in range(width)] for i in range(depth)])
        self.humidity = np.full((width, depth), fill_value=0.0)
        # self.humidity[0:int(width / 2), :] = np.full((int(width / 2), depth), fill_value=0.5)
        self.drop_dead()
        self.slope = self.calculate_slope()
        sea_blue = np.array([0, 102, 153, 256]) / 256
        sandy_brown = np.array([244, 164, 96, 256]) / 256
        self.sea_sand = mpl.colors.ListedColormap(np.concatenate((np.tile(sea_blue, 128).reshape(128, 4),
                                                                  np.tile(sandy_brown, 128).reshape(128, 4)), axis=0))
        hot = plt.cm.get_cmap('Oranges', 256)
        sea_wet_sand = hot(np.linspace(0, 1, 256))
        sea_wet_sand[0:160] = sea_wet_sand[256 - 160::]
        sea_wet_sand[160::] = np.tile(sea_blue, 256 - 160).reshape(256 - 160, 4)
        self.sea_wet_sand = mpl.colors.ListedColormap(sea_wet_sand)

        self.shear_rate = 1.5
        self.humidify_rate = 0.01

        self.delta_humidity = np.flip(np.exp(np.linspace(-10, 0, self.depth))) * self.humidify_rate
        self.delta_humidity = np.tile(self.delta_humidity, self.width).reshape(self.depth, self.width)
        self.delta_humidity = np.transpose(self.delta_humidity)

    def calculate_slope(self):
        slope = np.asarray([(self.edge[i + self.delta] - self.edge[i - self.delta]) / (2 * self.delta)
                            for i in range(self.delta, self.width - self.delta)])
        # for d in range(1, self.delta):
        #     slope[d: self.width - self.delta * 2 - d] = \
        #     np.asarray([((self.edge[i + d] - self.edge[i - d]) / (2 * d) + slope[i]) / 2
        #                 for i in range(d, self.width - self.delta * 2 - d)])
        slope = np.insert(slope, 0, np.full([self.delta, ], slope[0]))
        slope = np.append(slope, np.full([self.delta, ], slope[-1]))
        return np.arctan(slope)

    def update_life(self):
        self.life = np.array([[i >= self.edge[j] for j in range(self.width)] for i in range(self.depth)])

    def drop_dead(self):
        self.humidity = np.array(
            [[self.humidity[i, j] if self.life[i, j] else 1. for j in range(self.width)] for i in range(self.depth)])

    def bye_prick(self):
        for i in range(self.delta, self.width - self.delta):
            if self.edge[i] < min(self.edge[i - self.delta:i + self.delta]):
                self.edge[i] = max(self.edge[i - self.delta], self.edge[i + self.delta])
        # self.life = np.array([[i >= self.edge[j] for j in range(self.width)] for i in range(self.depth)])

    def wave(self):
        # Let's assume the slope and humidity level and the angle of the impact will change what our sand castle look like

        cosine = abs(np.cos(self.slope))
        sine = abs(np.sin(self.slope))

        # Then, according to the humidity level and angle of impact, we'll determine the san cell that is currently alive
        delta_shear = sine * self.shear_rate
        for k in range(self.delta):
            for i in range(self.width):
                # print("Current value of i is {}".format(i))
                # print("Current value of edge is {}".format(self.edge[i]))
                if (delta_shear[i] + 1 / (1 - self.humidity[self.edge[i], i]) - 1) > 1:
                    # self.life[self.edge[i], i] = False
                    self.edge[i] += 1 if self.edge[i] < self.depth - 1 else 0
                    delta_shear[i] -= delta_shear[i] / self.delta

        self.bye_prick()
        # We'll firstly update the humidity level according to the angle of impact
        self.delta_humidity *= cosine
        # print(delta_humidity.shape)
        # print(self.humidity.shape)
        for i in range(self.width):
            # print("Current i value is {}, and current edge value is {}".format(i, self.edge[i]))
            # print("Humidity from {} to {} is changed by {} to {} of delta humidity"
            #       .format(self.edge[i], self.depth, 0, self.depth - self.edge[i] - 1))
            self.humidity[self.edge[i]::, i] += self.delta_humidity[0:self.depth - self.edge[i], i]

        # Finally, we'll update the sand castle's humidity information and slope information
        self.slope = self.calculate_slope()
        # self.drop_dead()


# Please run them in interactive mode to get proper image output
# ipython is recommended
# from cell import *
# delta should be no smaller than 1
sim = SandCastleSimulator2D(1000, 1000, 5)
plt.figure()
plt.imshow(sim.humidity, cmap=sim.sea_wet_sand)
plt.figure()
plt.imshow(sim.life, cmap=sim.sea_sand)
sim.wave()
for _ in range(100):
    sim.wave()
sim.update_life()
sim.drop_dead()
plt.figure()
plt.imshow(sim.humidity, cmap=sim.sea_wet_sand)
