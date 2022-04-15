import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

# Question 4 - Metropolis Algorithm


def J(x):
    return x


if __name__ == "__main__":

    x_n = rd.randn()
    n = 0
    kT = 1
    N = 1000000
    M = 100000
    epsilon = .1
    stable_states = []

    while n < N:
        R = rd.choice([0, 1])
        x_hat = R

        deltaJ = J(x_hat) - J(x_n)

        q = np.exp(-deltaJ/kT)
        r = rd.uniform(0, 1)

        a = 0 if r > q else 1

        if deltaJ < 0:
            x_n = x_hat
        else:
            x_n = (1-a) * x_n + a * x_hat

        n += 1
        if n > N - M:
            stable_states.append(x_n)

    E_F_X = np.mean(stable_states)

    fig, ax1 = plt.subplots()
    _, _, bars = plt.hist(stable_states, density=True, bins=(np.arange(3)-0.5))
    plt.bar_label(bars)
    plt.xticks((0, 1))
    plt.show()


