import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt


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


    def transition(X):
        transition_vec = transition_matrix.T[X]
        r = rd.uniform(0, 1)
        if r < transition_vec[0]:
            return 0
        if r < transition_vec[0] + transition_vec[1]:
            return 1
        else:
            return 2


    X = 1
    for _ in range(3):
        X = transition(X)

    print("### ITEM b ###")
    print("Valor final: {} \n\n".format(X))


    ################# item c #################
    history = []
    episode = []

    for _ in range(100):
        X = rd.choice([0, 1, 2])
        episode.append(X)
        for _ in range(3):
            X = transition(X)
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

    val, vec = np.linalg.eig(transition_matrix)
    print("Matriz dos auto vetores:\n {}".format(vec))

