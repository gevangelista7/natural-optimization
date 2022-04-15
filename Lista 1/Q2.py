import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

# Question 2 - Metropolis Algorithm


def J(x):
    return (x - 3) ** 2


if __name__ == "__main__":
    T = 1
    X = [1, 2, 3, 4, 5]
    print('========== Item (a) ==========')
    for x in X:
        print("x: {} -> boltzmann(x): {}".format(x, np.exp(-J(x) / T)))

    boltzmann = [np.exp(-J(x) / T) for x in X]

    print('\n\n\n')
    print('========== Item (b) ==========')

    x_n = rd.choice(X)
    n = 0
    kT = 1
    N = 1000000
    M = 100000
    epsilon = 1
    stable_states = []
    transact = []
    # "small eps" = same magnitude of the smaller difference in state space

    while n < N:
        R = rd.choice((-1, 1))
        possible_x_npp = x_n + epsilon * R
        # possible_x_npp = possible_x_npp % max(X) + 1  # closed box
        if possible_x_npp == 6:
            possible_x_npp = 1
        if possible_x_npp == 0:
            possible_x_npp = 5

        deltaJ = J(possible_x_npp) - J(x_n)

        q = np.exp(-deltaJ/kT)
        r = rd.uniform(0, 1)

        a = 0 if r > q else 1

        if deltaJ < 0:
            x_n = possible_x_npp
        else:
            x_n = (1-a)*x_n + a*possible_x_npp

        n += 1

        if n > N - M:
            stable_states.append(x_n)


    E_F_X = np.mean(stable_states)
    print("E[F(X)] = {}".format(E_F_X))
    print("M: {} \nlen(stable_states): {}".format(M, len(stable_states)))
    print("Teste do tamanho da amostra válida: {}".format(len(stable_states) == M))

    # plt.hist(stable_states, bins=(np.arange(6)+0.5))
    _, _, bars = plt.hist(stable_states, density=True, bins=(np.arange(6)+0.5))
    plt.bar_label(bars)
    plt.show()





