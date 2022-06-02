from statistical_verification_torch import statistical_verification_torch
from math import ceil, log
from DataBase import DataBase
from SearchHeuristics import ParameterGridSearch
import numpy as np

NC = 50
P = 100

algo_res_database = DataBase("resultados_sensibilidade_NC50.csv")
stat_res_database = DataBase("resultados_vrf_stat_sensibilidade_NC50.csv")

search_heuristic = ParameterGridSearch(
    T0_list=[10 ** i for i in np.linspace(3, 7, 5)],
    Tmin_list=[1],
    alpha_list=list(np.linspace(.999, .99, 3)), #list(np.concatenate((np.linspace(.999, .99, 3), np.linspace(.95, .82, 3)))),
    max_iterations_list=[1],
    epsilon_list=[1e-3, 1e-1, 1],
    delta_list=[1e-2]
)

if __name__ == "__main__":
    print("NC = {}".format(NC))
    parameters = search_heuristic.new_parameters()
    eps, delta, Tmin, _, T0, alpha = parameters
    max_iterations = ceil(3 * log(Tmin / T0, alpha))
    parameters = list(parameters)
    parameters[3] = max_iterations

    while True:
        print('Combinações de parâmetros remanescentes: {}'.format(len(search_heuristic.combinations)))
        statistical_verification_torch(parameters, NC, P, 10, algo_res_database, stat_res_database)

        parameters = search_heuristic.new_parameters()
        if parameters is None:
            break
        eps, delta, Tmin, _, T0, alpha = parameters
        max_iterations = ceil(3 * log(Tmin/T0, alpha))
        parameters = list(parameters)
        parameters[3] = max_iterations

