import time
import numpy.random as rd
import numpy as np
from utils import generate_point_cloud_with_optimum
from SearchHeuristics import ParameterGridSearch
from DataBase import DataBase
from instrumented_DA_torch import instrumented_DA_torch
from statistical_verification_torch import statistical_verification_torch
from math import log, ceil
import torch

if __name__ == "__main__":
    original_seed = 0
    rd.seed(original_seed)
    NC = 14
    P = 100


    # níveis a serem analisados
    search_heuristic = ParameterGridSearch(
        T0_list=[10 ** i for i in np.linspace(2, 5, 4)],
        Tmin_list=[1],
        alpha_list=list(np.concatenate((np.linspace(.999, .99, 2), np.linspace(.95, .82, 3)))),
        max_iterations_list=[1],
        epsilon_list=[1e-6, 1e-4, 1e-1],
        delta_list=[1e-4, 1e-2]
    )
    parameters = search_heuristic.new_parameters()
    eps, delta, Tmin, _, T0, alpha = parameters
    max_iterations = ceil(3 * log(Tmin/T0, alpha))
    parameters = list(parameters)
    parameters[3] = max_iterations

    # data bases
    algo_res_database = DataBase("resultados_projeto1_exp7.csv")
    stat_res_database = DataBase("resultados_vrf_stat_exp7_01.csv")

    while NC <= 30:
        data_vectors, tgtJ, tgtD, _ = generate_point_cloud_with_optimum(n_clusters=NC, core_points=P, cores_dispersion=NC,
                                                                        dimension=2)
        data_vectors = torch.tensor(data_vectors, device='cuda', dtype=torch.float64)

        satisfactory_result = False

        while not satisfactory_result:
            print("{} Parâmetros para tentativa de solução com NC = {}: \n {}".format(time.asctime(), NC, parameters))
            results = instrumented_DA_torch(data_vectors=data_vectors, NC=NC, T0=T0, Tmin=Tmin,
                                            max_iterations=max_iterations, alpha=alpha, eps=eps, delta=delta,
                                            target_J=tgtJ, seed=original_seed)

            algo_res_database.data_entry(level_results=[results], fieldnames=results.keys())

            # controles dos parâmetros
            satisfactory_result = results['satisfactory_result']
            if satisfactory_result:
                print("><> {} :Possivel combinação com resultado satisfatório para NC={}: \n {}".format(time.asctime(),
                                                                                                        NC, parameters))
                success_rate = statistical_verification_torch(parameters=parameters, NC=NC, P=P, n_rodadas=20,
                                                              algo_res_database=algo_res_database,
                                                              stat_verify_database=stat_res_database)
                rd.seed(original_seed)
                print("Taxa de sucesso na verifição estatística: {}".format(success_rate))
                if success_rate < 0.75:
                    satisfactory_result = False
                else:
                    continue

            parameters = search_heuristic.new_parameters()
            if parameters is None:
                break
            else:
                eps, delta, Tmin, _, T0, alpha = parameters
                parameters = list(parameters)
                max_iterations = ceil(2 * log(Tmin / T0, alpha))
                parameters[3] = max_iterations



        if parameters is None:
            print("Varredura de parâmetros completa, sem combinação satisfatória para resolução com NC = {}".format(NC))
            break
        print("NC = {} concluído com sucesso!".format(NC))
        search_heuristic.new_T0_basis(T0)
        search_heuristic.reset_combinations()
        NC += 1

    instrumented_DA_torch(data_vectors=data_vectors, NC=NC-1, T0=T0, Tmin=Tmin, max_iterations=max_iterations,
                          alpha=alpha, eps=eps, delta=delta, target_J=tgtJ, history_mode=True, seed=original_seed)
