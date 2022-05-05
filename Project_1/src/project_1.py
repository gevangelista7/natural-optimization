import numpy as np
import numpy.random as rd
import time
from natural_optimizers import deterministic_annealing
from utils import generate_point_cloud


if __name__ == "__main__":
    rd.seed(0)

    T0 = 10
    Tmin = 0.1
    max_iterations = 200
    alpha = 0.95
    eps = 1e-6
    delta = 1e-3
    NC = 8

    J_last = 0
    Tf_last = Tmin
    level_completed = False
    results = {}
    T_condition_try_count = 0

    # níveis a ser analisado
    J_trigger = 10

    while NC < 20:
        data_vectors = generate_point_cloud(NC, 100)
        while not level_completed:
            start_time = time.time()
            Y, p_yx, i, history_J, history_D, history_T, _ = \
                deterministic_annealing(X=data_vectors, n_centroid=NC, T0=T0, Tmin=Tmin, max_iterations=max_iterations,
                                        alpha=alpha, epsilon=eps, delta=delta)

            # registro de resultados:
            elapsed_time = time.time() - start_time
            final_J = history_J[-1]
            final_D = history_D[-1]
            final_T = history_T[-1]

            results[(NC, T0, max_iterations, alpha, eps, delta)] = (Y, final_J, p_yx, final_D, final_T, elapsed_time)

            # controles dos parâmetros
            if final_T > Tmin:
                alpha *= 0.95
                T_condition_try_count += 1
                if T_condition_try_count == 2:
                    delta *= 1.01
                    T_condition_try_count = 0














        NC += 1

