from statistical_verification import statistical_verification
from math import ceil, log
from DataBase import DataBase

NC = 50
P = 100

algo_res_database = DataBase("res_busca_vrf_stat_local_np.csv")
stat_res_database = DataBase("res_stat_local_np.csv")

T0 = 100000
Tmin = 1
alpha = 0.999
eps = 1e-1
delta = 1e-1
max_iterations = ceil(3 * log(Tmin/T0, alpha))
parameters = eps, delta, Tmin, max_iterations, T0, alpha

if __name__ == "__main__":
    print("NC = {}".format(NC))
    statistical_verification(parameters, NC, P, 5, algo_res_database, stat_res_database)

