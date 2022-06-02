
import torch as t
import numpy as np
from .BinaryMutationVec import BinaryMutationVec
from .UniformCrossoverVec import UniformCrossoverVec
from .StochasticUniversalSampling import StochasticUniversalSampling
from .FitnessFunctionWithCounter import FitnessFunctionWithCounter
from .Register import GARegister, DataPrep


class StandardGeneticAlgorithmVec:
    def __init__(self, dimension, n_population, n_bit_mutate, mutation_rate, fitness_function, max_ite, tgt_fitness):
        t.no_grad()
        self.population = t.empty((n_population, dimension), dtype=t.bool, device='cuda')
        self.population.random_()

        self.offspring = t.empty((n_population, dimension), dtype=t.bool, device='cuda')
        self.offspring.random_()

        self.offspring_fitness = t.zeros(n_population)
        self.max_ite = max_ite
        self.tgt_fitness = tgt_fitness

        self.fitness_function = FitnessFunctionWithCounter(fitness_function, self.offspring_fitness)
        self.fitness_function.evaluate(self.offspring)

        self.mutation = BinaryMutationVec(population=self.population, pop_rate=mutation_rate, n_bit=n_bit_mutate, )
        self.recombination = UniformCrossoverVec(offspring=self.offspring, dimensions=dimension, parents=self.population)

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




