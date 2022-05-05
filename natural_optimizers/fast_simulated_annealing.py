import numpy as np
import numpy.random as rd


def fast_sa(calcJ, x0, N, K, epsilon, T0):
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

        x_hat = x_n + rd.standard_cauchy(size=len(x_n)) * epsilon
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
            x_n = rd.normal(size=10)
            T = T0 / (1 + k)
            if k > K:
                finished = True

    return x_min, J_min, history
