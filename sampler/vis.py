import matplotlib
matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
import numpy as np

class Visualizer:

    def plot_curve(self, x, data1, data2, error1, error2, name):
        fig = plt.figure()

        plt.grid(True)

        l1, = plt.plot(x, data1, label="P($g_1|H$)")
        l2, = plt.plot(x, data2, label="P($g_2|H$)")

        plt.legend(handles=[l1,l2])

        plt.xlabel('$\epsilon$')
        plt.ylabel('average belief of goals given the trajectory histories')

        plt.errorbar(x, data1, error1, fmt='-o', markersize=8, capsize=8)
        plt.errorbar(x, data2, error2, fmt='-o', markersize=8, capsize=8)

        plt.show()
        plt.savefig(name)

    def plot_single_curve(self, x, data, error, name):

        fig1 = plt.subplot(2, 1, 1)

        plt.grid(True)

        plt.plot(x, data, marker='o', markersize=8)

        plt.xlabel('$\epsilon$')
        plt.ylabel('average belief of true goal')

        fig2 = plt.subplot(2, 1, 2)

        plt.grid(True)

        plt.plot(x, error, color='orange', marker='o', markerfacecolor='orange', markersize=8)

        plt.xlabel('$\epsilon$')
        plt.ylabel('variance of true goal belief')

        plt.show()
        plt.savefig(name)

    def plot_traj(self, traj, name):
        plt.figure()
        plt.plot(traj)
        plt.grid(True)

        plt.xlabel('time step $\Delta t$')
        plt.ylabel('belief of goals given the trajectory histories P($g_1|H$)')

        plt.show()
        plt.savefig(name)

    def plot_trajs(self, trajs, name):
        plt.figure()

        plt.grid(True)

        plt.xlabel('time step $\Delta t$')
        plt.ylabel('belief of goals given the trajectory histories P($g_1|H$)')

        for traj in trajs:
            plt.plot(traj)

        plt.show()
        plt.savefig(name)

    def plot_trajs_compare(self, trajs, trajs_, name):

        fig1 = plt.subplot(2, 1, 1)

        plt.grid(True)

        for traj in trajs:
            plt.plot(traj)

        plt.xlabel('time step $\Delta t$ ($\epsilon = 0.00$)')
        plt.ylabel('belief of $g_1$ given a trajectory')

        fig2 = plt.subplot(2, 1, 2)

        plt.grid(True)

        for traj in trajs_:
            plt.plot(traj)

        plt.xlabel('time step $\Delta t$ ($\epsilon = 0.27$)')
        plt.ylabel('belief of $g_1$ given a trajectory')

        plt.show()
        plt.savefig(name)

    def plot_heat_map(self, size, S, decoys, goal, starting, name):
        folder = '../graph_config/'
        l, w = size[0], size[1]
        temp = np.zeros((l, w))
        for state in S:
            s = tuple(state)
            i, j = state[0] - 1, state[1] - 1
            if s in decoys:
                temp[i, j] = 1
                if s in goal:
                    temp[i,j] += 1
            if s == starting:
                temp[i, j] = 5

        plt.imshow(temp, interpolation='nearest')
        # cbar = fig.colorbar(cax)  # ticks=[-1, 0, 1]
        plt.savefig(name + ".png")

    def draw_grid(self):

        return


if __name__ == '__main__':
    fig = plt.figure()

    plt.grid(True)

    ax1 = [0, 1, 2, 3]
    ax2 = [2, 3, 2, 4]

    l1, = plt.plot(ax1, ax2, label="p1")

    plt.legend(handles=[l1])

    plt.xlabel('$\epsilon$')
    plt.ylabel('average belief of goals')

    plt.show()