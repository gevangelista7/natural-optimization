import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt


def deltaJ(to, origin, J):
    return J[to] - J[origin]


def f_boltz(to, origin, T, J):
    return np.exp(-deltaJ(to, origin, J)/T)


def acept_trans_prob(to, origin, T, J):
    return min(f_boltz(to, origin, T, J), 1.0)


def get_transition_matrix(T, J):
    l = len(J.items())
    tm = np.zeros((l, l))
    for j in range(len(tm)):
        for i in range(len(tm[0])):
            if (i == (j+1) % l) or (i == (j-1) % l):
                tm[i, j] = 0.5 * acept_trans_prob(i+1, j+1, T, J)
        tm[j, j] = 1 - sum(tm[:, j])

    return tm


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
    M5 = get_transition_matrix(5, J)

    val, vec = np.linalg.eig(M5)
    idx_unit = np.where(np.around(val, 1) == 1)[0].item()
    vec_prob_M5 = vec.T[idx_unit]
    vec_prob_M5 = vec_prob_M5/sum(vec_prob_M5)

    print("M_5 = \n{}".format(M5))
    print("PI_5 = {}".format(vec_prob_M5))

    M10 = get_transition_matrix(10, J)
    val, vec = np.linalg.eig(M5)
    idx_unit = np.where(np.around(val, 1) == 1)[0].item()
    vec_prob_M10 = vec.T[idx_unit]
    vec_prob_M10 = vec_prob_M10/sum(vec_prob_M10)
    print("M_10 = \n{}".format(M10))
    print("PI_10 = {}".format(vec_prob_M10))

    print(" #### item c ####")
    min_M5 = M5[2, 1]
    min_M10 = M10[2, 1]
    # transição 2 -> 3, que é a transição de maior diferença de energia
    # ou seja deltaJmax = Jmax - Jmin
    # T aumenta a probabilidade da transição ocorrer
    # min_M10/min_M5 = exp(-deltaJ/T10)/exp(-deltaJ/T5)
    # min_M10/min_M5 = exp(deltaJ(-1/T10 + 1/T5))



