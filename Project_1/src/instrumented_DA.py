import time
from natural_optimizers import deterministic_annealing
from utils import plot_curves_da, plot_points_da


def instrumented_DA(data_vectors, NC, T0, Tmin, max_iterations, alpha, eps, delta, target_D, seed,
                    history_mode=False):
    start_time = time.time()
    Y, p_yx, i_final, history_J, history_D, history_T, _ = \
        deterministic_annealing(X=data_vectors, n_centroid=NC, T0=T0, Tmin=Tmin, max_iterations=max_iterations,
                                alpha=alpha, epsilon=eps, delta=delta)
    elapsed_time = time.time() - start_time

    # critério de solução
    satisfactory_result = history_D[i_final - 1] < target_D

    # registro de resultados:
    results = {
        'NC': NC,
        'satisfactory_result': satisfactory_result,
        'T0': T0,
        'max_iterations': max_iterations,
        'alpha': alpha,
        'eps': eps,
        'delta': delta,
        'i_final': i_final,
        'final_J': history_J[i_final - 1],
        'final_D': history_D[i_final - 1],
        'final_T': history_T[i_final - 1],
        'elapsed_time': elapsed_time,
        'target_D': target_D,
        'seed': seed
    }

    history = {
        'history_J': history_J,
        'history_D': history_D,
        'history_T': history_T,
        'Y': Y,
        'p_yx': p_yx
    }

    if history_mode:
        plot_curves_da(history_J, history_D, history_T, i_final, "NC = {}, P = {} ".format(NC, int(len(data_vectors)/NC)))
        plot_points_da(data_vectors, Y, "NC = {}".format(NC), with_voronoi=True)
        return results, history

    return results
