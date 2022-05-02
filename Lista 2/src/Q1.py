import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from utils import transition, invariant_vector

if __name__ == "__main__":
    transition_matrix = np.array([[0.50, 0.25, 0.25],
                                  [0.25, 0.50, 0.25],
                                  [0.25, 0.25, 0.50]])

    p0 = np.array([[.3, .4, .3]]).T

    tm_3 = np.linalg.matrix_power(transition_matrix, 3)

    p_3 = np.matmul(tm_3, p0)

    print("### ITEM b ###")
    print("P(3) = \n{}\n\n".format(p_3))

    ################# item b #################
    p0 = np.array([[0.0, 1.0, 0.0]]).T

    print("### ITEM b ###")
    X = 1
    print("{}o estado: {}".format(1, X))
    for i in range(3):
        X = transition(X, transition_matrix)
        print("{}o estado: {}".format(i + 2, X))

    print("Valor final: {} \n\n".format(X))


    ################# item c #################
    print("### ITEM c ###")
    history = []
    episode = []

    for _ in range(100):
        X = rd.choice([0, 1, 2])
        episode.append(X)
        for _ in range(3):
            X = transition(X, transition_matrix)
            episode.append(X)

        history.append(episode)
        episode = []

    history = np.array(history)
    print("History shape: {}".format(history.shape))
    print("History max: {}".format(history.max()))
    print("History min: {}".format(history.min()))

    ################# item d #################
    fig, ax = plt.subplots(1, 4, sharey=True)
    fig.axes[0].set_ylabel("Ocorrências")
    for i in range(len(history.T)):
        x = history.T[i]
        bars_i = ax[i].hist(x, bins=(np.arange(4)-0.5))
        ax[i].bar_label(bars_i[-1], [int(l) for l in bars_i[0]])
        ax[i].set_title("X{}".format(i))
        ax[i].set_xticks((0, 1, 2))

    PI = invariant_vector(transition_matrix)
    print("Vetor invariante de probabilidades: \n {}".format(PI))
    print("Para 100 iterações observamos que o algoritmo não necessariamente converge para o esperado em comparação "
          "com o vetor invariante.")

