import time

import numpy.random as rd
import numpy as np
from utils import generate_point_cloud
from SearchHeuristics import ParameterGridSearch
from DataBase import DataBase
from instrumented_DA import instrumented_DA
from statistical_verification import statistical_verification

if __name__ == "__main__":
    original_seed = 0
    rd.seed(original_seed)
    NC = 17
    P = 100


    # níveis a serem analisados
    search_heuristic = ParameterGridSearch(
        T0_list=[10**i for i in np.linspace(1, 4, 4)],
        Tmin_list=[1],
        alpha_list=list(np.linspace(.97, .82, 4)),
        max_iterations_list=list(np.linspace(1000, 10000, 10)), #np.concatenate((np.linspace(100, 1000, 10), np.linspace(2000, 10000, 9))),
        epsilon_list=[1e-6],
        delta_list=[1e-3]
    )
    parameters = search_heuristic.new_parameters()
    eps, delta, Tmin, max_iterations, T0, alpha = parameters

    # data bases
    algo_res_database = DataBase("results/resultados_projeto1_exp4.csv")
    stat_res_database = DataBase("results/resultados_vrf_stat_exp4.csv")

    while NC <= 50:
        data_vectors, real_mean_dist, std_dev = generate_point_cloud(core_number=NC, core_points=P, cores_dispersion=NC,
                                                                     dimension=2)
        final_D_criteria = real_mean_dist + 2 * std_dev
        satisfactory_result = False
        level_results = []
        while not satisfactory_result:
            print("{} Parâmetros para tentativa de solução com NC = {}: \n {}".format(time.asctime(), NC, parameters))
            results = instrumented_DA(data_vectors=data_vectors, NC=NC, T0=T0, Tmin=Tmin, max_iterations=max_iterations,
                                      alpha=alpha, eps=eps, delta=delta, target_D=final_D_criteria, seed=original_seed)

            level_results.append(results)

            # controles dos parâmetros
            satisfactory_result = results['satisfactory_result']
            if satisfactory_result:
                print(">>> {} :Possivel combinação com resultado satisfatório para NC={}: \n {}".format(time.asctime(),
                                                                                                      NC, parameters))
                success_rate = statistical_verification(parameters, NC, P, algo_res_database, stat_res_database,
                                                        n_rodadas=1)
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
                eps, delta, Tmin, max_iterations, T0, alpha = parameters

            algo_res_database.data_entry(level_results=level_results, fieldnames=results.keys())

        if parameters is None:
            print("Varredura de parâmetros completa, sem combinação satisfatória para resolução com NC = {}".format(NC))
            break
        print("NC = {} concluído com sucesso!".format(NC))
        search_heuristic.new_T0_basis(T0)
        search_heuristic.new_max_ite_basis(max_iterations)
        search_heuristic.reset_combinations()
        NC += 1

    instrumented_DA(data_vectors=data_vectors, NC=NC-1, T0=T0, Tmin=Tmin, max_iterations=max_iterations,
                    alpha=alpha, eps=eps, delta=delta, target_D=final_D_criteria, history_mode=True, seed=original_seed)
