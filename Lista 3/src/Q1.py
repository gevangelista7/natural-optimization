from deterministic_annealing import deterministic_annealing
from utils import plot_curves, plot_points
import numpy as np


data_vector = np.array([[5, 4], [4, 5], [5, 5], [-5, -4], [-4, -5], [0, 0], [1, 1], [-1, -1]])

np.random.seed(0)
Y10, p_yx10, i10, history_J10, history_D10, history_T10, history_Y10 = \
    deterministic_annealing(X=data_vector, n_centroid=2, T0=100, Tmin=0.1, max_iterations=300,
                            alpha=.9, epsilon=1e-6, delta=1e-3)


print(" A matriz de probabilidades P_ygivenx obtida foi:\n{}".format(p_yx10))
print("Os pontos centroides ficaram localizados nos pontos: \n{}".format(Y10))
print("O algoritmo foi completou a solução em {} iterações.".format(i10))
print("A figura a seguir apresenta a configruação final do problema. ")
plot_points(data_vector, Y10, "T=10")
plot_curves(history_J10, history_D10, history_T10, i10, "T=10")

### item b ###
Y01, p_yx01, i01, history_J01, history_D01, history_T01, history_Y01 = \
    deterministic_annealing(X=data_vector, n_centroid=2, T0=0.1, Tmin=0.05, max_iterations=300, alpha=.995,
                            epsilon=1e-6, delta=1e-3)


print(" A matriz de probabilidades P_ygivenx obtida foi:\n{}".format(p_yx01))
print("Os pontos centroides ficaram localizados nos pontos: \n{}".format(Y01))
print("O algoritmo foi completou a solução em {} iterações.".format(i01))
print("A figura a seguir apresenta a configruação final do problema. ")
plot_points(data_vector, Y01, "T=0.1")
plot_curves(history_J01, history_D01, history_T01, i01, "T=0.1")

