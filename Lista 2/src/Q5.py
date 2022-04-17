import numpy as np
import numpy.random as rd
from Q2 import acept_trans_prob
import matplotlib.pyplot as plt


J = {
    (0, 0): 0.2,
    (0, 1): 0.3,
    (1, 0): 0.3,
    (1, 1): 0.1
}

# decodificando os estados como números binários:
# alteração do bit x1 equivale a +- 1 e do bit x2 equivale a +- 2
J = {
    0: 0.2,
    1: 0.3,
    2: 0.3,
    3: 0.1
}


def get_transition_matrix(T, J, passo):
    l = len(J.items())
    tm = np.zeros((l, l))
    for j in range(len(tm)):
        for i in range(len(tm[0])):
            j_hat1 = (j+passo) % l
            if i == j_hat1:
                tm[i, j] = acept_trans_prob(i, j, T, J)
        tm[j, j] = 1 - sum(tm[:, j])

    return tm


if __name__ == "__main__":
    T = 0.5
    print("#### item a ####")
    tm_x1 = get_transition_matrix(T, J, 1)
    print("Matriz de transição com modificação do bit x1")
    print(tm_x1)

    val, vec = np.linalg.eig(tm_x1)
    idx_unit_x1 = np.where(np.around(val, 1) == 1)[0].item()
    vec_prob_x1 = vec.T[idx_unit_x1]
    vec_prob_x1 = vec_prob_x1/sum(vec_prob_x1)

    print("Vetor de probabilidades invariante para perturbações em x1:\n {}".format(vec_prob_x1))

    tm_x2 = get_transition_matrix(T, J, 2)
    print("Matriz de transição com modificação do bit x2")
    print(tm_x2)

    # val, vec = np.linalg.eig(tm_x2)
    # idx_unit_x2 = np.where(np.around(val, 1) == 1)[0].item()
    # vec_prob_x2 = vec.T[idx_unit_x2]
    # vec_prob_x2 = vec_prob_x2/sum(vec_prob_x2)
    #
    # print("Vetor de probabilidades invariante para perturbações em x1:\n {}".format(vec_prob_x2))

    print("#### item b ####")
    T = .5
    fb = []
    for j in J.values():
        fb.append(np.exp(-j/T))

    fb = np.array(fb)
    fb_norm = fb/sum(fb)
    print("Fatores de Boltzmann para cada estado: \n {}".format(fb))

    print("#### item c ####")
    # Simulated Annealing
    x_n = rd.choice([0, 1, 2, 3])
    x_min = x_n
    J_n = J[x_n]
    J_min = J_n
    N = 10000
    K = 4
    k = 1

    T_0 = 1
    T = T_0

    n = 0
    finished = False
    history = []

    """ Otimização por simulated annealing"""
    while not finished:
        n += 1

        x_hat = x_n + rd.choice([1, 2])
        x_hat = x_hat % (max(J.keys())+1)

        J_possible = J[x_hat]

        deltaJ = J_possible - J_n
        r = rd.uniform(0, 1)

        if r > np.exp(deltaJ / T):
            x_n = x_hat
            J_n = J_possible
            if J_n < J_min:
                J_min = J_n
                x_min = x_n

        history.append((x_min, J_min))
        if n % N == 0:
            x_n = rd.choice([0, 1, 2, 3])
            T = T_0 / np.log2(1 + k)
            k += 1
            if k > K:
                finished = True

    print("X_min = ", x_min, "\nJ_min = ", J_min)
    plt.plot(history)

