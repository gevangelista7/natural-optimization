import numpy as np
import numpy.random as rd
from deterministic_annealing import deterministic_annealing
from utils import plot_points, plot_curves
import time


np.random.seed(1)
P = 100
NC = 10
cluster_centers = np.random.normal(0, 1, [NC, 2])
data_vector = []
for center in cluster_centers:
    data_vector.append(center)
    for _ in range(P-1):
        data_vector.append(center + rd.normal(0, .1, 2))

data_vector = np.array(data_vector)
T = 10
start_time = time.time()
Y, p_yx, i, history_J, history_D, history_T, _ = deterministic_annealing(X=data_vector, n_centroid=8, T0=T,
                                                                         Tmin=0.1, max_iterations=300, alpha=.9,
                                                                         epsilon=1e-6, delta=1e-3)
print("time elapsed: {:.2f}s".format(time.time() - start_time))

plot_points(data_vector, Y, "T={}".format(T))
plot_curves(history_J, history_D, history_T, i, "T={}".format(T))



