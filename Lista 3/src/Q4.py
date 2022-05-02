from matplotlib import pyplot as plt

from deterministic_annealing import deterministic_annealing
from utils import plot_curves, plot_points
import numpy as np


if __name__ == "__main__":
    data_vector = np.array(((1, 2), (2, 1), (-2, -1), (-1, -2)))
    X = data_vector
    T = 5
    Y0 = np.array(((1, 1), (-1, -1)))
    d_xy = np.zeros((len(Y0), len(X)))

    for i_x in range(len(X)):
        for i_y in range(len(Y0)):
            d_xy[i_y, i_x] = np.sum((X[i_x] - Y0[i_y]) ** 2)

    p_yx = np.exp(-d_xy / T)
    Z_x = np.sum(p_yx, axis=0)
    p_yx = p_yx / Z_x

    Y1 = np.zeros(Y0.shape)
    for i_y in range(len(Y0)):
        Y1[i_y] = np.dot(p_yx[i_y], X) / np.sum(p_yx[i_y])

    print("### item c ###")
    d = 1

    T_var = np.linspace(1e-4, 200, 1000)
    Y_t = np.array([3/2*np.tanh(6*d/t) for t in T_var])

    fig, ax = plt.subplots(1, 2)


    ax[0].plot(T_var, Y_t)
    ax[1].plot(T_var, Y_t)
    ax[1].set_xlim((0, 4))
    ax[0].set_xlabel("T")
    ax[1].set_xlabel("T")
    ax[0].set_ylabel("d[2]")
    ax[0].grid()
    ax[1].grid()
    ax[1].annotate(Y_t[0], (T_var[0], Y_t[0]))
    ax[1].annotate("{:.5}".format(Y_t[10]), (T_var[10], Y_t[10]))
    plt.show()




