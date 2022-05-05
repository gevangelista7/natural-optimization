import numpy as np
import numpy.random as rd


def simulated_annealing(calcJ, disturb, x0, N, K, T0):
    x_n = x0
    x_min = x_n
    J_n = calcJ(x_n)
    J_min = J_n
    k = 1
    T = T0

    n = 0
    finished = False
    history = []

    while not finished:
        n += 1

        # x_hat = x_n + epsilon * rd.normal(0, 1, x_n.shape)
        x_hat = disturb(x_hat)
        J_hat = calcJ(x_hat)

        _deltaJ = J_hat - J_n
        r = rd.uniform(0, 1)

        if r < np.exp(-_deltaJ / T):
            x_n = x_hat
            J_n = J_hat
            if J_n < J_min:
                J_min = J_n
                x_min = x_n

        history.append((x_min, J_min, J_n))
        if n % N == 0:
            k += 1
            x_n = disturb(np.zeros(x0.shape))
            T = T0 / np.log2(2 + k)
            if k > K:
                finished = True

    return x_min, J_min, history
