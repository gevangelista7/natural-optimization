import time
from natural_optimizers import deterministic_annealing_torch
from utils import plot_curves_da, plot_points_da
import torch

device = 'cuda' if torch.has_cuda else 'cpu'


def instrumented_DA_torch(data_vectors, NC, T0, Tmin, max_iterations, alpha, eps, delta, target_J, seed,
                          history_mode=False):
    start_time = time.time()
    Y, p_yx, i_final, history_J, history_D, history_T, _ = \
        deterministic_annealing_torch(X=data_vectors, n_centroid=NC, T0=T0, Tmin=Tmin, max_iterations=max_iterations,
                                      alpha=alpha, epsilon=eps, delta=delta)
    elapsed_time = time.time() - start_time

    # critério de solução
    satisfactory_result = abs(history_J[i_final - 1] - target_J) < 0.15*abs(target_J)

    # registro de resultados:
    results = {
        'NC': NC,
        'satisfactory_result': satisfactory_result.item(),
        'T0': T0,
        'max_iterations': max_iterations,
        'alpha': alpha,
        'eps': eps,
        'delta': delta,
        'i_final': i_final,
        'target_J': target_J.item(),
        'final_J': history_J[i_final - 1].item(),
        'final_D': history_D[i_final - 1].item(),
        'final_T': history_T[i_final - 1].item(),
        'elapsed_time': elapsed_time,
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
        plot_curves_da(history_J.cpu(), history_D.cpu(), history_T.cpu(), i_final, "NC = {}, P = {} "
                       .format(NC, 100))    # int(len(data_vectors)/NC)))
        plot_points_da(data_vectors.cpu(), Y.cpu(), "NC = {}".format(NC), with_voronoi=True)
        return results, history

    return results
