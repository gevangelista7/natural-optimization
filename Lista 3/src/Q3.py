import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from test_functions import generate_rastrigin
from simulated_annealing import simulated_annealing
from fast_simulated_annealing import fast_sa


if __name__ == "__main__":
    rd.seed(0)
    dimensions = 10
    x0 = rd.normal(size=dimensions)
    calcJ = generate_rastrigin(dimensions)

    # calcJ = generate_rosenbrock(1, 100)
    # calcJ = styblinski_tang

    results = {}
    i = 0
    N_list = [1e3, 1e4, 1e5]
    K_list = [8, 16, 32, 64]
    epsilon_list = [1e-1, 1e-2, 1e-3]
    T0_list = [1e-2, 1e-1, 1e1, 1e2]
    for N in N_list:
        for K in K_list:
            for eps in epsilon_list:
                for T0 in T0_list:
                    i += 1
                    x_min_sa, J_min_sa, history_sa = simulated_annealing(calcJ=calcJ, x0=x0, N=N, K=K, epsilon=eps,
                                                                         T0=T0)
                    norm_x_sa = np.linalg.norm(x_min_sa)
                    results[('SA', N, K, eps, T0)] = (J_min_sa, x_min_sa, norm_x_sa, history_sa)

                    x_min_fsa, J_min_fsa, history_fsa = fast_sa(calcJ=calcJ, x0=x0, N=N, K=K, epsilon=eps, T0=T0)
                    norm_x_fsa = np.linalg.norm(x_min_sa)
                    results[('FSA', N, K, eps, T0)] = (J_min_fsa, x_min_fsa, norm_x_fsa, history_fsa)

                    print("{} tentativas realizadas".format(i))
                    if norm_x_fsa < 0.3:
                        print("Candidata a solução válida achada pelo FSA!")
                        print("X_min = {} (norma= {}) \nJ_min = {}".format(x_min_fsa, norm_x_fsa, J_min_fsa))
                    if norm_x_sa < 0.3:
                        print("Candidata a solução válida achada pelo SA!")
                        print("X_min = {} (norma= {}) \nJ_min = {}".format(x_min_sa, norm_x_sa, J_min_sa))

    # fig, ax = plt.subplots(2)
    # ax[0].plot([data[-1] for data in history_fsa])
    # ax[1].plot([data[1] for data in history_fsa])


