import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt




def deltaJ(to, origin):
    return J[to] - J[origin]


def f_boltz(to, origin, T):
    return np.exp(-deltaJ(origin, to)/T)


def acept_trans_prob(to, origin, T):
    return min(f_boltz(origin, to, T), 1.0)


def get_transition_matrix(T, J):
    l = len(J.items())
    tm = np.zeros((l, l))
    for j in range(len(tm)):
        for i in range(len(tm[0])):
            if (i == (j+1) % l) or (i == (j-1) % l):
                tm[i, j] = 0.5 * acept_trans_prob(i+1, j+1, T)
        tm[j, j] = 1 - sum(tm[:, j])

    return tm




if __name__ == "__main__":
    T = 1

    J = {
        1: .3,
        2: .1,
        3: .1,
        4: .2
    }

    tm = get_transition_matrix(T, J)
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
    tm = get_transition_matrix(T, J)
    print("#### item a ####")
    print("Problema modificado, proposto na lista 2 enviada em 14ABR22:")
    print(tm)

    print("#### item b ####")

    print("#### item c ####")
    print("O vetor invariante corresponde ao autovetor "
          "\"normalizado\" para somar 1, associado ao autovalor"
          "unitário")

    val, vec = np.linalg.eig(tm)
    print(val)
    print(vec)

    idx_unitario = np.where(np.around(val, 1) == 1)[0].item()
    vec_prob = vec.T[idx_unitario]
    vec_prob = vec_prob/sum(vec_prob)

    print("Vetor invariante associado a matriz de transição M do item a: v = \n {}".format(vec_prob))

    print("#### item d ####")






