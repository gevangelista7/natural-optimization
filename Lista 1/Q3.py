import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

# Question 3 - Metropolis Algorithm


def J(x):
    return x ** 2


if __name__ == "__main__":

    x_n = rd.randn()
    n = 0
    kT = .1
    N = 100000
    M = 100000
    epsilon = .1
    stable_states = []

    while n < N:
        R = rd.uniform(-100, 100)
        possible_x = x_n + epsilon * R

        deltaJ = J(possible_x) - J(x_n)

        q = np.exp(-deltaJ/kT)
        r = rd.uniform(0, 1)

        a = 0 if r > q else 1

        if deltaJ < 0:
            x_n = possible_x
        else:
            x_n = (1-a) * x_n + a * possible_x

        n += 1
        if n > N - M:
            stable_states.append(x_n)

    E_F_X = np.mean(stable_states)
    # print("E[F(X)] = {}".format(E_F_X))
    fig, ax1 = plt.subplots()
    _, _, bars = plt.hist(stable_states, density=False, bins=100)

    """TESTE DE ADERÊNCIA"""
    X = np.linspace(-1, 1, 101)
    boltz = np.exp(-X**2/.1)

    ax2 = ax1.twinx()
    plt.plot(X, boltz/sum(boltz), 'g-')

    plt.show()


