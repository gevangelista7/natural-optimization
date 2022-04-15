import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt


def J(x):
    return -x + 100 * (x - 0.2) ** 2 * (x - 0.8) ** 2


if __name__ == "__main__":
    x_n = 0.0
    x_min = x_n
    J_n = J(x_n)
    J_min = J_n
    N = 10000
    K = 8
    k = 1

    T_0 = 10
    T = T_0
    epsilon = 5e-1
    n = 0
    finished = False
    history = []

    while not finished:
        n += 1

        x_possible = x_n + epsilon * rd.randn()
        J_possible = J(x_possible)

        deltaJ = J_possible - J_n
        r = rd.uniform(0, 1)

        if r > np.exp(deltaJ/T):
            x_n = x_possible
            J_n = J_possible
            if J_n < J_min:
                J_min = J_n
                x_min = x_n

        history.append((x_min, J_min))
        if n % N == 0:
            k += 1
            T = T_0 / np.log2(1 + k)
            if k == K:
                finished = True


    print("X_min = {:.2} \nJ_min = {:.4}".format(x_min, J_min))

    X = np.linspace(0, 1, 100)
    plt.plot(X, [J(x) for x in X])



