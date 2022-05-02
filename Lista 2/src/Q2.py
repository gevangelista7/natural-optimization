import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from utils import transition_matrix, transition, invariant_vector

if __name__ == "__main__":
    T = 1

    J = {
        1: .3,
        2: .1,
        3: .1,
        4: .2
    }

    tm = transition_matrix(T, J)
    print("Problema original, resolvido em sala:")
    print(tm)

    J = {
        1: .5,
        2: .2,
        3: .3,
        4: .1,
        5: .4
    }

    T = 0.1
    tm = transition_matrix(T, J)
    print("#### item a ####")
    print("Problema modificado, proposto na lista 2 enviada em 14ABR22:")
    print(tm)

    print("#### item b ####")
    X = 1
    print("{}o estado: {}".format(1, X))
    for i in range(3):
        X = transition(X, tm)
        print("{}o estado: {}".format(i+2, X))

    print("#### item c ####")
    print("O vetor invariante corresponde ao autovetor "
          "\"normalizado\" para somar 1, associado ao autovalor"
          "unitário")

    vec_prob = invariant_vector(tm)

    print("Vetor invariante associado a matriz de transição M do item a: v = \n {}".format(vec_prob))

    print("#### item d ####")
    fb = []
    T = .1
    for j_i in J.values():
        fb.append(np.exp(-j_i/T))

    fb = np.array(fb)
    fb_norm = fb/sum(fb)
    print("Vetor dos fatores de Boltzmann: \n{}".format(fb))
    print("Vetor dos fatores de Boltzmann normalizado: \n{}".format(fb_norm))
    print("Diferenças entre os fatores de Boltzmann e o "
          "vetor de probabilidades: \n{}".format(vec_prob - fb_norm))
    print("Portanto os vetores são iguais")

    print("#### item e ####")
    # Metropolis Algorithm

    Ts = [0.100, 0.0631, 0.0500, 0.0431, 0.0387, 0.0356, 0.0333, 0.0315, 0.0301, 0.0289]
    idx_T = 0
    T = Ts[idx_T]

    x_n = rd.choice(list(J.keys()))
    n = 0
    kT = Ts[0]
    N = 1000 * len(Ts)
    M = N
    epsilon = 1
    stable_states = []

    while n < N:
        R = rd.choice((-1, 1))
        x_hat = x_n + epsilon * R
        if x_hat == 6:
            x_hat = 1
        if x_hat == 0:
            x_hat = 5

        deltaJ = J[x_hat] - J[x_n]

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

        if n % 1000 == 0:
            if n == N:
                continue
            else:
                idx_T += 1
                T = Ts[idx_T]

    plt.hist(stable_states, bins=(np.arange(len(J.keys())+1)+.5))
    plt.scatter(list(J.keys()), vec_prob*N, c='red')


