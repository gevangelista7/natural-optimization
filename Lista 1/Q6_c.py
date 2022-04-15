import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt


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

    kTs = [.001, .01, .1, 1, 10, 100]
    N = 100000
    M = 10000
    epsilon = 5e-2
    E_F_Xs = []

    for kT in kTs:
        x_n = rd.randn(5, 2)  # configuração inicial qualquer aleatória
        stable_states = []
        n = 0
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

        print("kT = {} computado".format(kT))
        E_F_Xs.append(np.mean([L(x) for x in stable_states]))

    plt.plot(kTs, E_F_Xs)
