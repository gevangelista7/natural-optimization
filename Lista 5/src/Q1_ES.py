
import torch as t
from itertools import product
from random import randint
from GeneticAlgorithm import EvolutionStrategy, FitnessFunctionWithCounter
from GeneticAlgorithm import GARegister
from ClusteringFitness import ClusteringFitness
from OptimTestFunctions import ackley_t_inv
from utils import generate_point_cloud_with_optimum, plot_points_da


# n_cluster :      Number of clusters
# n_points:        Number of points per cluster
# n_pop:           Number of individuals in population
# dim:             Number of dimensions

# Y.shape:          ( n_cluster , dim )   >>  genotype
# popY.shape:       ( n_pop, n_cluster, dim )
# X.shape:          ( n_cluster * n_points, dim )

# idv.shape:        ( 1, n_cluster * n_dim )
# population.shape: ( n_pop, n_cluster * n_dim * 2 )

t.set_grad_enabled(False)

class AckleyFitness(FitnessFunctionWithCounter):
    def __init__(self):
        super(AckleyFitness, self).__init__()

    def fitness_update(self):
        super(AckleyFitness, self).fitness_update()
        self.fitness_array.copy_(ackley_t_inv(self.evaluation_population).unsqueeze(1))

if __name__ == '__main__':

    mu_list = [20]
    lambda_mu_list = [25]
    max_eval = 5e4
    dim = 20
    n_rounds = 150
    tolerance = -.85

    algo_name = 'ES'
    common_path = '../res/' + algo_name + "_Ackley_{}".format(dim)

    results_registry = GARegister(algo_name="ES",
                                  filename="ES_Ackley_{}".format(dim),
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
            fitness_function = AckleyFitness()

            algo = EvolutionStrategy(individual_dimension=dim,
                                     fitness_function=fitness_function,
                                     tgt_fitness=tolerance,
                                     max_eval=max_eval,
                                     _eps0=1e-3,
                                     _lambda=_lambda,
                                     _mu=_mu,
                                     until_max_eval=True,
                                     seed=seed,
                                     dirname=common_path+'/lambda{}mu{}'.format(_lambda, _mu),
                                     filename='Rodada_{}'.format(i)
                                     )


            result = algo.run()
            results_registry.data_entry([result])

            i += 1










