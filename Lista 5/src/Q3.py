
from GeneticAlgorithm import FitnessFunctionWithCounter, BinaryMutationInt, StochasticUniversalSampling, GARegister, \
    DataPrep
import torch as t
import numpy as np

from UniformCrossoverInt import UniformCrossoverInt


class StandardGeneticAlgorithm:
    def __init__(self, dimension, n_population, n_bit_mutate, fitness_function, max_ite, tgt_fitness):
        self.population = t.randint(0, 2**dimension-1, size=(n_population,))
        self.offspring = t.randint(0, 2**dimension-1, size=(n_population,))
        self.offspring_fitness = t.zeros(n_population)
        self.max_ite = max_ite
        self.tgt_fitness = tgt_fitness

        self.fitness_function = FitnessFunctionWithCounter(fitness_function, self.offspring_fitness)
        self.fitness_function.evaluate(self.offspring)

        self.mutation = BinaryMutationInt(population=self.population, pop_rate=.5, dimension=dimension,
                                          n_bit=n_bit_mutate, )
        self.recombination = UniformCrossoverInt(offspring=self.offspring, dimensions=dimension, parents=self.population)

        self.survivors_selection = StochasticUniversalSampling(offspring_fitness=self.offspring_fitness,
                                                               selected=self.population, candidates=self.offspring)
        self.register = GARegister(algo_name='SGA', data_header=['iter_n', 'iter_time', 'gen_best_fit', 'gen_best_idv',
                                                                 'gen_mean_fit', 'gen_worst_fit'])
        self.data_processor = DataPrep(population=self.offspring, fitness=self.offspring_fitness)

    def run(self):
        final_result = {
            'max_fit': -np.inf,
            'best_idv':  None,
            'final_gen_mean_fit': None,
            'final_gen_best_fit': None,
            'final_gen_best_idv': None
        }

        while self.fitness_function.counter < self.max_ite:
            self.recombination.execute()
            self.mutation.execute()
            self.fitness_function.evaluate(self.offspring)
            self.survivors_selection.execute()

            iter_result = self.data_processor.processed_result(self.fitness_function.counter)
            self.register.data_entry([iter_result])

            if iter_result['gen_best_fit'] > final_result['max_fit']:
                final_result['max_fit'] = iter_result['gen_best_fit']
                final_result['best_idv'] = iter_result['gen_best_idv']

            if t.mean(self.offspring_fitness) >= self.tgt_fitness:
                break

        final_result['final_gen_mean_fit'] = t.mean(self.offspring_fitness)
        final_result['final_gen_best_fit'] = t.max(self.offspring_fitness)
        final_result['final_gen_best_idv'] = self.offspring[t.argmax(self.offspring_fitness)]

        return final_result


class FonsecaFlemingFitness(FitnessFunctionWithCounter):
    def __init__(self, n_dim, n_bit, n_pop, device):
        super(FonsecaFlemingFitness, self).__init__()
        # self.n_pop = n_pop
        self.n_dim = n_dim
        self.n_bit = n_bit
        self.device = device

        self.min_val = -4
        self.max_val = 4

        self.fitness_array = None
        self.evaluation_population = None

        self.bin_arrays = None
        self.decoded_values = None

        self.decode_mask = 2 ** t.arange(self.n_bit, device=device)

    def link(self, evaluation_population: t.Tensor, fitness_array: t.Tensor):
        super(FonsecaFlemingFitness, self).link(evaluation_population=evaluation_population,
                                                fitness_array=fitness_array)
        self.n_pop = evaluation_population.shape[0]
        self.bin_arrays = self.evaluation_population.view(self.n_pop, self.n_dim, self.n_bit)

    def decode_values(self):
        pow2 = self.bin_arrays * self.decode_mask
        pos = t.sum(pow2, dim=-1)
        self.decoded_values = self.min_val + (self.max_val - self.min_val) * pos / 2 ** self.n_bit

