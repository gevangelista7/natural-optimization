import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.spatial import Voronoi, voronoi_plot_2d


def plot_points(data_vectors, Y, title, with_voronoi=False, Yvor=None, xlim=(None, None), ylim=(None, None)):
    if with_voronoi:
        if Yvor is None:
            raise('No Yvor was received')
        else:
            vor = Voronoi(Yvor)
            fig = voronoi_plot_2d(vor)
    plt.figure()
    plt.title(title)
    plt.plot(data_vectors[:, 0], data_vectors[:, 1], 'k.', label="X (data vector)")
    plt.plot(Y[:, 0], Y[:, 1], 'r.', markersize=20, label="Y")
    if xlim[0] is not None:
        plt.xlim(xlim)
    if ylim[0] is not None:
        plt.ylim(ylim)
    plt.grid()
    plt.show()


def plot_curves(history_J, history_D, history_T, final_i, title):
    history_J = history_J[:final_i]
    history_D = history_D[:final_i]
    history_T = history_T[:final_i]

    plt.figure()
    plt.title(title)
    plt.plot(-history_J, 'r-', label='- Cost function')
    plt.plot(history_D, 'k-', label='Mean Distance')
    plt.plot(history_T, 'b-', label='Temperature')
    plt.legend()
    plt.grid()
    plt.show()

