import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from copy import deepcopy


### considerando os vértices de 0 a 7 e suas posições xyz como o binário de seu identificador:
# def calcJ(config):
#     pass
#     centroids = calcCentroids(config)
#     total_j = 0
#     for ponto in range(8):
#         p = ponto2coord(ponto)
#         if config & 2**ponto == 2**ponto:
#             total_j += np.linalg.norm(p - centroids[1])**2
#         else:
#             total_j += np.linalg.norm(p - centroids[0])**2
#     return total_j
#
#
# def ponto2coord(ponto):
#     p = np.zeros(3)
#     for dim in range(3):
#         p[dim] = (ponto & 2 ** dim) / 2 ** dim
#     return p
#
# def calcCentroids(config):
#     centroids_pos = [np.zeros(3), np.zeros(3)]
#     centroids_count = [0, 0]
#     for ponto in range(8):
#         if config & 2**ponto == 2**ponto:
#             for dim in range(3):
#                 if ponto & 2 ** dim == 2 ** dim:
#                     centroids_pos[1][dim] += 1
#             centroids_count[1] += 1
#         else:
#             for dim in range(3):
#                 if ponto & 2**dim == 2**dim:
#                     centroids_pos[0][dim] += 1
#             centroids_count[0] += 1
#
#     if centroids_count[0] != 0:
#         centroids_pos[0] /= centroids_count[0]
#     if centroids_count[1] != 0:
#         centroids_pos[1] /= centroids_count[1]
#
#     return centroids_pos
# #
ponto2coord = {
    1: np.array((0, 0, 1)),
    2: np.array((0, 1, 1)),
    3: np.array((1, 0, 1)),
    4: np.array((1, 1, 1)),
    5: np.array((0, 0, 0)),
    6: np.array((0, 1, 0)),
    7: np.array((1, 0, 0)),
    8: np.array((1, 1, 0))
}


def calcCentroids(state):
    centroids_pos = [np.zeros(3), np.zeros(3)]
    centroids_count = [0, 0]
    for point in range(1, 9):
        if state[point-1] == 1:
            centroids_pos[1] += ponto2coord[point]
            centroids_count[1] += 1
        else:
            centroids_pos[0] += ponto2coord[point]
            centroids_count[0] += 1
    if centroids_count[0] != 0:
        centroids_pos[0] /= centroids_count[0]
    if centroids_count[1] != 0:
        centroids_pos[1] /= centroids_count[1]

    return centroids_pos


def calcJ(state):
    centroids = calcCentroids(state)
    total_J = 0
    for point in range(1, 9):
        if state[point-1] == 0:
            total_J += np.linalg.norm(ponto2coord[point] - centroids[0])**2
        if state[point-1] == 1:
            total_J += np.linalg.norm(ponto2coord[point] - centroids[1])**2
    return total_J


def disturb(state):
    site = rd.choice(8)
    s = deepcopy(state)
    s[site] = (s[site] + 1) % 2
    return s


if __name__ == "__main__":
    x_n = rd.choice(2, 8)
    x_min = x_n
    J_n = calcJ(x_n)
    J_min = J_n
    N = 10000
    K = 12
    k = 1

    T_0 = 1
    T = T_0

    n = 0
    finished = False
    history = []

    while not finished:
        n += 1

        x_hat = disturb(x_n)
        J_possible = calcJ(x_hat)

        _deltaJ = J_possible - J_n
        r = rd.uniform(0, 1)

        if r < np.exp(- _deltaJ / T):
            x_n = x_hat
            J_n = J_possible
            if J_n < J_min:
                J_min = J_n
                x_min = x_n

        history.append((x_min, J_min, J_n))
        if n % N == 0:
            k += 1
            x_n = rd.choice(2, 8)
            T = T_0 / np.log2(1 + k)
            if k > K:
                finished = True

    print("X_min = ", x_min, "\nJ_min = ", J_min)
    plt.plot([data[-1] for data in history])

    print("\n#### item b ####")
    T = 1.0

    e01 = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    ef1 = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    dJ1 = calcJ(ef1) - calcJ(e01)
    prob_trans_1 = np.exp(-dJ1/T)
    print("A probabilidade da transição do estado {} para o estado {} é \n {}"
          .format(e01, ef1, prob_trans_1))

    e02 = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    ef2 = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    dJ2 = calcJ(ef2) - calcJ(e02)
    prob_trans_2 = np.exp(-dJ2/T)
    print("A probabilidade da transição do estado {} para o estado {} é \n {}"
          .format(e02, ef2, prob_trans_2))

    print("\n#### item c ####")
    T = 0.1
    custos = np.array([4.0, 4.5, 4.53, 4.67, 5.0, 5.14, 5.33, 5.5, 5.6, 6.0])
    n_estados = np.array([6, 8, 48, 24, 24, 16, 24, 24, 64, 18])
    f_boltz = np.exp(-custos)
    Z = sum(f_boltz * n_estados)
    est = np.array([0, 0, 0, 0, 1,  1, 1, 1])
    prob = np.exp(-calcJ(est))/Z
    print("Probabilidade de ocorrência de x = {}: {:.4}".format(est, prob))

