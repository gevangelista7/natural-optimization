import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from utils import  invariant_vector, transition_matrix

J = {
    1: 7,
    2: 1,
    3: 10,
    4: 4
}

if __name__ == "__main__":
    print(" #### item a ####")
    """Otimização por Simulated Annealing"""

    x_n = rd.choice([1, 2, 3, 4])
    x_min = x_n
    J_n = J[x_n]
    J_min = J_n
    N = 10000
    K = 6
    k = 1

    T_0 = 1
    T = T_0

    n = 0
    finished = False
    history = []

    while not finished:
        n += 1

        x_hat = x_n + rd.choice((-1, 1))
        if x_hat == 5:
            x_hat = 1
        if x_hat == 0:
            x_hat = 4

        J_possible = J[x_hat]

        _deltaJ = J_possible - J_n
        r = rd.uniform(0, 1)

        if r > np.exp(_deltaJ / T):
            x_n = x_hat
            J_n = J_possible
            if J_n < J_min:
                J_min = J_n
                x_min = x_n

        history.append((x_min, J_min))
        if n % N == 0:
            k += 1
            x_n = rd.choice([1, 2, 3, 4])
            T = T_0 / np.log2(1 + k)
            if k > K:
                finished = True

    print("X_min = ", x_min, "\nJ_min = ", J_min)

    print(" #### item b ####")
    M5 = transition_matrix(5, J)
    vec_prob_M5 = invariant_vector(M5)

    print("M_5 = \n{}".format(M5))
    print("PI_5 = {}".format(vec_prob_M5))

    M10 = transition_matrix(10, J)
    # val, vec = np.linalg.eig(M5)
    # idx_unit = np.where(np.around(val, 1) == 1)[0].item()
    # vec_prob_M10 = vec.T[idx_unit]
    # vec_prob_M10 = vec_prob_M10/sum(vec_prob_M10)
    vec_prob_M10 = invariant_vector(M10)
    print("M_10 = \n{}".format(M10))
    print("PI_10 = {}".format(vec_prob_M10))

    print(" #### item c ####")
    min_M5 = M5[2, 1]
    min_M10 = M10[2, 1]

    deltaJmax = 9  # Jmax - Jmin
    N = 2  # número de transições possíveis
    T5 = 5
    T10 = 10
    p5 = 1 / N * np.exp(-deltaJmax / T5)
    p10 = 1 / N * np.exp(-deltaJmax / T10)

    # p5/p10 = 1/N*np.exp(-deltaJmax*(1/T5 - 1/T10))
    # p5/p10 = np.exp(-deltaJmax*(2/T10 - 1/T10)) = np.exp(-deltaJmax*1/T10) = N*p10

    # RESPOSTA:
    # p5 = N * p10 ** 2
    print("A relação entre as probabilidade e onúmero de transições possíveis é dado por: p5 = N * p10 ** 2 ")



