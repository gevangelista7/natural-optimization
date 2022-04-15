import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt


def J(x):
    return (
        x[0] ** 4 +
        x[1] ** 4 +
        x[2] ** 4 +
        x[3] ** 4
    ) ** 0.25


if __name__ == "__main__":
    x_n = np.array([1, 2, 3, 4])
    x_min = x_n
    J_n = J(x_n)
    J_min = J_n
    N = 10000
    K = 32
    k = 1

    T_0 = 1
    T = T_0
    epsilon = 5e-3

    n = 0
    finished = False
    history = []

    while not finished:
        n += 1

        x_possible = x_n + epsilon * rd.randn(4)
        J_possible = J(x_possible)

        deltaJ = J_possible - J_n
        r = rd.uniform(0, 1)

        if r > np.exp(deltaJ / T):
            x_n = x_possible
            J_n = J_possible
            if J_n < J_min:
                J_min = J_n
                x_min = x_n

        history.append((x_min, J_min))
        if n % N == 0:
            k += 1
            T = T_0 / np.log2(1 + k)
            if k > K:
                finished = True


    res_str = "X_min = "
    for i in x_min:
        res_str += f"{i:.4}" + ", "

    res_str += f"\nJ_min = {J_min:.4}"

    print(res_str)



