
import torch as t
from itertools import product
from random import randint
from GeneticAlgorithm import EvolutionStrategy
from GeneticAlgorithm import GARegister
from ClusteringFitness import ClusteringFitness
from utils import generate_point_cloud_with_optimum, plot_points_da


# n_cluster :      Number of clusters
# n_points:        Number of points per cluster
# n_pop:           Number of individuals in population
# dim:             Number of dimensions

# Y.shape:          ( n_cluster , dim )   >> genotype
# popY.shape:       ( n_pop, n_cluster, dim )
# X.shape:          ( n_cluster * n_points, dim )

# idv.shape:        ( 1, n_cluster * n_dim )
# population.shape: ( n_pop, n_cluster * n_dim * 2 )

t.set_grad_enabled(False)


if __name__ == '__main__':

    mu_list = [10, 30, 90]
    lambda_mu_list = [7, 10, 25]
    max_eval = 5e5
    dim = 2
    n_rounds = 35
    tolerance = .05

    n_clusters = 10
    core_points = 100

    algo_name = 'ES'
    common_path = '../res/' + algo_name + "_NC_{}".format(n_clusters)

    results_registry = GARegister(algo_name="ES",
                                  filename="ES_NC_{}".format(n_clusters),
                                  dir_name=common_path,
                                  data_header=[
                                      'n_gen',
                                      'success',
                                      'best_fit',
                                      'eval_first_sol',
                                      'tgt_fit',
                                      'elapsed_time',
                                      'final_eval_counter',
                                      'final_gen_mean_fit',
                                      'final_gen_best_fit',
                                      'final_gen_worst_fit',
                                      'best_idv',
                                      'final_gen_best_idv',
                                      'seed',
                                      
                                      'lambda',
                                      'mu'
                                  ])

    for params in product(mu_list, lambda_mu_list):
        _mu = params[0]
        _lambda = _mu * params[1]

        i = 0
        while i < n_rounds:
            seed = randint(0, 1e6)
            X, minJ, minD, centers = generate_point_cloud_with_optimum(n_clusters=n_clusters,
                                                                       core_points=core_points,
                                                                       cores_dispersion=n_clusters,
                                                                       dimension=dim,
                                                                       T=1)
            X = t.tensor(X, device='cuda')
            X_limit = t.max(abs(X))
            fitness_function = ClusteringFitness(X=X,
                                                 n_clusters=n_clusters,
                                                 T=1)

            algo = EvolutionStrategy(individual_dimension=n_clusters * dim,
                                   fitness_function=fitness_function,
                                   tgt_fitness=- (1+tolerance) * minJ,
                                   max_eval=max_eval,
                                   _eps0=1e-3,
                                   _lambda=_lambda,
                                   _mu=_mu,
                                   until_max_eval=True,
                                   seed=seed,
                                   dirname=common_path+'/lambda{}mu{}'.format(_lambda, _mu),
                                   filename='Rodada_{}'.format(i),
                                   x_lim=(-X_limit, X_limit))


            result = algo.run()
            results_registry.data_entry([result])

            i += 1










