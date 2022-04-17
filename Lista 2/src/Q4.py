import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from Q2 import get_transition_matrix, acept_trans_prob, deltaJ


if __name__ == "__main__":

    J = {
        1: .3,
        2: .1,
        3: .1,
        4: .2
    }

    fb_t1 = []
    fb_t01 = []
    for v in J.values():
        fb_t01.append(np.exp(-v / 0.1))
        fb_t1.append(np.exp(-v / 1))

    fb_t1 = np.array(fb_t1)
    fb_t01 = np.array(fb_t01)

    print("Valores dos fatores de Boltzmann (T = 1.0): \n{}".format(fb_t1))
    print("Valores dos fatores de Boltzmann (T = 0.1): \n{}".format(fb_t01))


    print(" #### item c ####")
    tm_t1  = get_transition_matrix(1.0, J)
    tm_t01 = get_transition_matrix(0.1, J)

    print("Matriz de transição para T = 1.0")
    print(tm_t1)

    val, vec = np.linalg.eig(tm_t1)
    idx_unit = np.where(np.around(val, 1) == 1)[0].item()
    vec_prob_t1 = vec.T[idx_unit]
    vec_prob_t1 = vec_prob_t1 / sum(vec_prob_t1)
    fb_t1_norm = fb_t1 / sum(fb_t1)

    print("Vetor invariante (T=1.0): \n {}".format(vec_prob_t1))
    print("Vetor dos fatores de Boltzmann (T = 1.0): \n{}".format(fb_t1))
    print("Vetor dos fatores de Boltzmann normalizado: \n{}".format(fb_t1))
    print("Diferenças entre os fatores de Boltzmann e o "
          "vetor de probabilidades: \n{}".format(vec_prob_t1 - fb_t1_norm))
    print("Portanto os vetores são iguais")


    print("Matriz de transição para T = 0.1")
    print(tm_t01)

    val, vec = np.linalg.eig(tm_t01)
    idx_unit = np.where(np.around(val, 1) == 1)[0].item()
    vec_prob_t01 = vec.T[idx_unit]
    vec_prob_t01 = vec_prob_t01 / sum(vec_prob_t01)
    fb_t01_norm = fb_t01 / sum(fb_t01)

    print("Vetor invariante (T=0.1): \n {}".format(vec_prob_t01))
    print("Vetor dos fatores de Boltzmann (T = 0.1): \n{}".format(fb_t01))
    print("Vetor dos fatores de Boltzmann normalizado: \n{}".format(fb_t01))
    print("Diferenças entre os fatores de Boltzmann e o "
          "vetor de probabilidades: \n{}".format(vec_prob_t01 - fb_t01_norm))
    print("Portanto os vetores são iguais")


