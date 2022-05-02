import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from test_functions import generate_rastrigin
from fast_simulated_annealing import fsa


if __name__ == "__main__":
    rd.seed(0)
    x0 = rd.normal(size=8)

    # calcJ = generate_rosenbrock(1, 100)
    calcJ = generate_rastrigin(5)
    # calcJ = styblinski_tang

    x_min, J_min, history = fsa(calcJ=calcJ, x0=x0, N=1e4, K=8, epsilon=5e-2, T0=1e-2)

    print("X_min = {} (norma= {}) \nJ_min = {}".format(x_min, np.linalg.norm(x_min), J_min))
    fig, ax = plt.subplots(2)
    ax[0].plot([data[-1] for data in history])
    ax[1].plot([data[1] for data in history])


