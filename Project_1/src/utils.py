import numpy as np
import numpy.random as rd

def deltaJ(to, origin, J):
    return J[to] - J[origin]


def f_boltz(to, origin, T, J):
    return np.exp(-deltaJ(to, origin, J)/T)


def accept_trans_prob(to, origin, T, J):
    return min(f_boltz(to, origin, T, J), 1.0)


def transition_matrix(T, J):
    l = len(J.items())
    tm = np.zeros((l, l))
    for j in range(len(tm)):
        for i in range(len(tm[0])):
            if (i == (j+1) % l) or (i == (j-1) % l):
                tm[i, j] = 0.5 * accept_trans_prob(i + 1, j + 1, T, J)
        tm[j, j] = 1 - sum(tm[:, j])

    return tm

def transition_matrix_passo(T, J, passo):
    l = len(J.items())
    tm = np.zeros((l, l))
    for j in range(len(tm)):
        for i in range(len(tm[0])):
            j_hat1 = (j+passo) % l
            if i == j_hat1:
                tm[i, j] = accept_trans_prob(i, j, T, J)
        tm[j, j] = 1 - sum(tm[:, j])

    return tm


def invariant_vector(tm):
    val, vec = np.linalg.eig(tm)
    idx_unit = np.where(np.around(val, 1) == 1)[0].item()
    vec_prob = vec.T[idx_unit]
    vec_prob = vec_prob/sum(vec_prob)
    return vec_prob.real


def transition(X, transition_matrix):
    transition_vec = transition_matrix.T[X]
    r = rd.uniform(0, 1)
    for i in range(len(transition_vec)):
        if r < sum(transition_vec[:i+1]):
            return i