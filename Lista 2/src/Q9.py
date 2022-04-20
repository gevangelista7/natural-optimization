import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from Q2 import acept_trans_prob


def get_transition_matrix(T, J):
    l = len(J.items())
    tm = np.zeros((l, l))
    for j in range(len(tm)):
        for i in range(len(tm[0])):
            if (i == (j+1) % l) or (i == (j-1) % l):
                tm[i, j] = 0.5 * acept_trans_prob(i+1, j+1, T, J)
        tm[j, j] = 1 - sum(tm[:, j])

    return tm

def get_invariant_vector(tm):
    val, vec = np.linalg.eig(tm)
    idx_unit = np.where(np.around(val, 1) == 1)[0].item()
    vec_prob = vec.T[idx_unit]
    vec_prob = vec_prob/sum(vec_prob)

    return vec_prob

J = {
    1: 4,
    2: 1,
    3: 3,
    4: 2,
    5: 4,
}

if __name__ == "__main__":
    T1 = 1/np.log(2)
    tm_t1 = get_transition_matrix(T1, J)
    pi_t1 = get_invariant_vector(tm_t1)
    print("T1 = 1/ln(2)")
    print("M_T1 = \n{}".format(tm_t1))
    print("PI_T1 = {}".format(pi_t1))

    T2 = 1 / np.log(3)
    tm_t2 = get_transition_matrix(T2, J)
    pi_t2 = get_invariant_vector(tm_t2)
    print("T2 = 1/ln(3)")
    print("M_T2 = \n{}".format(tm_t2))
    print("PI_T2 = {}".format(pi_t2))
