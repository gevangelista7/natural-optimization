import numpy.random as rd
import torch

from utils import generate_point_cloud_with_optimum
from DataBase import DataBase
from instrumented_DA import instrumented_DA
from instrumented_DA_torch import instrumented_DA_torch
from math import ceil, log

if __name__ == "__main__":
    rd.seed(0)
    P = 100
    NC = 17

    data_vectors, minJ, tgtD, _ = generate_point_cloud_with_optimum(n_clusters=NC, core_points=P, cores_dispersion=NC, dimension=2)
    device = 'cuda'
    targetJ = minJ
    T0 = 10000000
    Tmin = 1
    alpha = 0.999
    max_iterations = ceil(2 * log(Tmin/T0, alpha))
    eps = 1e-6
    delta = 1e-3

    database = DataBase("exp1_vrf_curva.csv")

    if device == 'cuda':
        data_vectors = torch.tensor(data_vectors, device=device, dtype=torch.float64)
        results, history = instrumented_DA_torch(data_vectors=data_vectors, NC=NC, T0=T0, Tmin=Tmin,
                                                 max_iterations=max_iterations, alpha=alpha, eps=eps, delta=delta,
                                                 target_J=final_D_criteria, history_mode=True, seed=0)
    else:
        results, history = instrumented_DA(data_vectors=data_vectors, NC=NC, T0=T0, Tmin=Tmin,
                                           max_iterations=max_iterations, alpha=alpha, eps=eps, delta=delta,
                                           target_D=final_D_criteria, history_mode=True, seed=0)


    results['seed'] = 0

    database.data_entry(level_results=[results], fieldnames=results.keys())
    # success_rate = statistical_verification((eps, delta, Tmin, max_iterations, T0, alpha), NC=NC, P=P, n_rodadas=20,
    #                                         algo_res_database=database,
    #                                         stat_verify_database=DataBase("exp1_vrf_curva_stat.csv"))

    print("Final D = {}".format(results['final_D']))

