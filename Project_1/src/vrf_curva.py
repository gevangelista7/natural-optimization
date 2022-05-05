import numpy as np
import numpy.random as rd
from natural_optimizers import deterministic_annealing
from utils import generate_point_cloud, plot_curves_da, plot_points_da

if __name__ == "__main__":
    P = 100
    NC = 20
    data_vectors = generate_point_cloud(core_number=NC, core_points=P, cores_dispersion=NC, dimension=2)
    T0 = 150
    Tmin = 0.1  # exp max antes do inf = 709
    max_iterations = 300
    alpha = 0.95
    eps = 1e-6
    delta = 1e-3

    Y, p_yx, i, history_J, history_D, history_T, _ = \
        deterministic_annealing(X=data_vectors, n_centroid=NC, T0=T0, Tmin=Tmin,
                                max_iterations=max_iterations, alpha=alpha, epsilon=eps, delta=delta)

    plot_curves_da(history_J, history_D, history_T, i, "NC = {}, P = {} ".format(NC, P))
    plot_points_da(data_vectors, Y, "NC = {}".format(NC), with_voronoi=True)
