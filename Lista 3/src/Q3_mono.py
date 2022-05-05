import numpy as np
import numpy.random as rd
from test_functions import generate_rastrigin
import time
import matplotlib.pyplot as plt
from simulated_annealing import simulated_annealing

rd.seed(0)
dimensions = 20
x0 = rd.normal(size=dimensions)
calcJ = generate_rastrigin(20)

N = 1e6
K = 16
eps = 1e-1
T0 = 1e0
general_start_time = time.time()

start_time_sa = time.time()
x_min_sa, J_min_sa, history_sa = simulated_annealing(calcJ=calcJ, x0=x0, N=N, K=K, epsilon=eps, T0=T0)
elapsed_time_sa = time.time() - start_time_sa
norm_x_sa = np.linalg.norm(x_min_sa)

print("X_min = {} (norma(X)= {}) \nJ_min = {} \nTempo decorrido: {}".format(x_min_sa, norm_x_sa, J_min_sa, elapsed_time_sa))

# fig, ax = plt.subplots(2)
# ax[0].plot([data[-1] for data in history_sa])
# ax[1].plot([data[1] for data in history_sa])
