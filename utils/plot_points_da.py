import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d


def plot_points_da(data_vectors, Y, title, with_voronoi=False, Yvor=None, xlim=(None, None), ylim=(None, None)):
    if with_voronoi:
        if Yvor is None:
            Yvor = Y
        vor = Voronoi(Yvor)
        fig = voronoi_plot_2d(vor)
    else:
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
