import numpy as np
import numpy.random as rd
from utils import transition_matrix_passo, invariant_vector
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
    0: 0.2,     # 00
    1: 0.3,     # 01
    2: 0.3,     # 10
    3: 0.1      # 11
}

if __name__ == "__main__":
    T = 0.5
    print("#### item a ####")
    tm_x1 = transition_matrix_passo(T, J, 1)
    print("Matriz de transição com modificação do bit x1")
    print(tm_x1)

    tm_x2 = transition_matrix_passo(T, J, 2)
    print("Matriz de transição com modificação do bit x2")
    print(tm_x2)

    print("#### item b ####")
    T = .5
    fb = []
    for j in J.values():
        fb.append(np.exp(-j/T))

    fb = np.array(fb)
    fb_norm = fb/sum(fb)
    print("Fatores de Boltzmann para cada estado: \n {}".format(fb))
    print("Fatores de Boltzmann para cada estado normalizados: \n {}".format(fb_norm))

    print("M_x1 * vec_fb - vec_fb: {}".format(np.matmul(tm_x1, fb) - fb))
    print("M_x2 * vec_fb - vec_fb: {}".format(np.matmul(tm_x2, fb) - fb))
    print("Portanto o vetor dos fatores de Boltzmann se aproxima do invariável porém apresenta algum desvio")

    # tm = .5*tm_x1 + .5*tm_x2
    # vec_prob = invariant_vector(tm)
    # print("Vetor de probabilidades invariante considerando, considerando a escolha do bit perturbado"
    #       " equiprovável : \n {}".format(vec_prob))

    print("#### item e ####")
    T = 0.1
    tm_x1_t01 = transition_matrix_passo(T, J, 1)
    tm_x2_t01 = transition_matrix_passo(T, J, 2)
    tm_t01 = .5*tm_x1_t01 + .5*tm_x2_t01

    vec_prob_t01 = invariant_vector(tm_t01)
    p_J03 = vec_prob_t01[1] + vec_prob_t01[2]
    print("Pelo vetor de probabilidades invariantes temos que a probabilidade de x1x2 = 01 ou x1x2 = 10:\n"
          "{}".format(p_J03))



