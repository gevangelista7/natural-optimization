import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap


def J(X):
    res = 0.0
    for i in range(len(X)):
        res += np.linalg.norm(X[i]) ** 2
        for j in range(i + 1, len(X)):
            res += 1 / np.linalg.norm(X[i] - X[j]) ** 2

    return res


def L(X):
    return np.mean([np.linalg.norm(x) for x in X])

if __name__ == "__main__":

    x_n = rd.randn(5, 2)  # configuração inicial qualquer aleatória
    n = 0
    kT = .1
    N = 1000000
    M = 100000
    epsilon = 5e-2
    stable_states = []

    while n < N:
        R = rd.randn(5, 2)
        possible_x = x_n + epsilon * R

        deltaJ = J(possible_x) - J(x_n)

        q = np.exp(-deltaJ / kT)
        r = rd.uniform(0, 1)

        a = 0 if r > q else 1

        if deltaJ < 0:
            x_n = possible_x
        else:
            x_n = (1 - a) * x_n + a * possible_x

        n += 1
        if n > N - M:
            stable_states.append(x_n)

    X, Y = [], []
    for state in stable_states:
        for particle in state:
            X.append(particle[0])
            Y.append(particle[1])

    fig, axs = plt.subplots()
    axs.set_aspect('equal')
    plt.hexbin(X, Y, bins=1000)

    E_F_X = np.mean([L(x) for x in stable_states])
