import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print(" #### item a ####")
    T = 1
    J00 = 1

    # M1 00 -> 10:
    # 1/3 = exp(-(J10 - J00)/T)
    # ln(1/3) = -J10 + 1
    J10 = 1 - np.log(1/3)

    # M2 10 -> 11:
    # 1/3 = exp(-(J11 - J10)/T)
    # ln(1/3) = -J11 + J10
    J11 = J10 - np.log(1/3)

    # M1 01 -> 11:
    # 1/3 = exp(-(J11 - J01)/T)
    # ln(1/3) = -J11 + J01
    J01 = np.log(1/3) + J11

    print("A partir das transições M1 00 -> 10, M2 10 -> 11, e M1 01 -> 11, temos que:")
    print("J00 = {}".format(J00))
    print("J01 = {}".format(J01))
    print("J11 = {}".format(J11))
    print("J10 = {}".format(J10))

    print(" #### item b ####")
    M1 = np.array([[2 / 3, 0, 0, 1],
                   [0, 2 / 3, 1, 0],
                   [0, 1 / 3, 0, 0],
                   [1 / 3, 0, 0, 0]])

    M2 = np.array([[2 / 3, 1, 0, 0],
                   [1 / 3, 0, 0, 0],
                   [0, 0, 0, 1 / 3],
                   [0, 0, 1, 2 / 3]])

    M = .5 * M1 + .5 * M2

    print("A matriz existe p=0.5 de se alterar cada bit. Portanto a matriz de transição de cada passo "
          "deve ser composta de maneira que M = .5 * M1 + .5 * M2. Desta forma temos que M = \n{}".format(M))

    print(" #### item c ####")
    val, vec = np.linalg.eig(M)
    idx_unit = np.where(np.around(val, 1) == 1)[0].item()
    vec_prob = vec.T[idx_unit]
    vec_prob = vec_prob/sum(vec_prob)
    print("Vetor invariante normalizado: PI = {}".format(vec_prob))
    print("M1 * PI - PI: {}".format(np.matmul(M1, vec_prob) - vec_prob))
    print("M2 * PI - PI: {}".format(np.matmul(M2, vec_prob) - vec_prob))

    print("Comparativo de probabilidades: ")
    fb = np.array([np.exp(-J00), np.exp(-J01), np.exp(-J11), np.exp(-J10)])
    fb_norm = fb/sum(fb)

    plt.scatter(range(4), fb_norm, c='blue', marker='o', label="Boltzmann norm.")
    plt.scatter(range(4), vec_prob, c='red', marker='x', label="Vetor invariante")
    plt.xticks(range(4), ['00', '01', '11', '10'])
    plt.title("Vetor Invariante x Fatores de Boltzmann normalizados")


