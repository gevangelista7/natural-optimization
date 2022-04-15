import numpy as np
import numpy.random as rd
from math import modf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Question 5 - Metropolis Algorithm


def J(x):
    return (x[0] ** 2 + x[1] ** 2)


def mirror(x):
    if abs(x) < 1:
        return x
    else:
        f, w = modf(x)
        return w - f


if __name__ == "__main__":

    x_n = rd.randn(2)
    n = 0
    kT = 1
    N = 1000000
    M = 100000
    epsilon = 1e-1
    stable_states = []

    while n < N:
        R = rd.randn(2)
        x_hat = x_n + epsilon * R
        x_hat = np.array([mirror(x_k) for x_k in x_hat])

        deltaJ = J(x_hat) - J(x_n)

        q = np.exp(-deltaJ / kT)
        r = rd.uniform(0, 1)

        a = 0 if r > q else 1

        if deltaJ < 0:
            x_n = x_hat
        else:
            x_n = (1 - a) * x_n + a * x_hat

        n += 1
        if n > N - M:
            stable_states.append(x_n)

    # E = np.mean([x[0] ** 2 + x[1] ** 2
    #              if (abs(x[0]) < 1 and abs(x[1]) < 1)
    #              else 0
    #              for x in stable_states])

    E = np.mean([x[0] ** 2 + x[1] ** 2 for x in stable_states])

    Z = 4 * np.mean([np.exp(-(x[0] ** 2 + x[1] ** 2)) for x in rd.uniform(0, 1, (100000, 2))])


    print("Valor esperado da integral: {:.4}".format(E * Z))

    # X = [x[0] for x in stable_states]
    # Y = [x[1] for x in stable_states]
    # plt.hexbin(X,Y)
    # plt.scatter(X, Y, marker='.')
    # plt.show()

    # from matplotlib import cm
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # pontos = rd.uniform(0, 1, (1000, 2))
    # X = np.array([x[0] for x in pontos])
    # Y = np.array([x[1] for x in pontos])
    # X, Y = np.meshgrid(X, Y)
    # Z = np.exp(-(X ** 2 + Y ** 2))
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
