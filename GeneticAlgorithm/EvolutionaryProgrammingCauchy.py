import numpy as np
import torch as t
from .MutationMPLCauchy import MutationMPLCauchy
from .StochasticUniversalSampling import StochasticUniversalSampling
from .Register import GARegister, DataPrep, FinalResultProcessor


class EvolutionaryProgrammingCauchy:
    def __init__(self, individual_dimension, fitness_function, tgt_fitness, max_ite, _lambda, _mu, filename,
                 _tau1=None, _tau2=None, _eps0=None):
        self.device = "cuda" if t.has_cuda else "cpu"
        t.no_grad()

        self.dimension = individual_dimension
        self.tgt_fitness = tgt_fitness
        self.max_ite = max_ite
        self._mu = _mu
        self._lambda = _lambda

        self._eps0 = 1e-3 if _eps0 is None else _eps0
        self._tau1 = 1 / np.sqrt(2 * individual_dimension) if _tau1 is None else _tau1
        self._tau2 = 1 / np.sqrt(2 * np.sqrt(individual_dimension)) if _tau2 is None else _tau2

        self.population = t.normal(0, 1, (self._mu, self.dimension * 2), device=self.device)

        self.offspring = t.ones((self._lambda, self.dimension * 2), device=self.device)
        self.x_values_c = self.offspring[:, :self.dimension]

        self.fitness_function = fitness_function
        self.offspring_fitness = t.normal(0, 1, (_lambda, 1), device=self.device)

        self.mutation = MutationMPLCauchy(population=self.population, offspring=self.offspring, _eps0=self._eps0,
                                          _tau1=self._tau1, _tau2=self._tau2, device=self.device)
        self.survivors_selection = StochasticUniversalSampling(offspring_fitness=self.offspring_fitness,
                                                               selected=self.population, candidates=self.offspring)
        self.register = GARegister(filename=filename, algo_name='EP', data_header=['iter_n', 'iter_time', 'gen_best_fit', 'gen_best_idv',
                                                                                   'gen_mean_fit', 'gen_worst_fit'])
        self.data_processor = DataPrep(population=self.offspring, fitness=self.offspring_fitness)

        self.final_result_registry = FinalResultProcessor(offspring_fitness=self.offspring_fitness,
                                                      tgt_fitness=self.tgt_fitness)

    def run(self):
        while self.fitness_function.counter < self.max_ite:
            self.mutation.execute()
            self.offspring_fitness[:, :] = self.fitness_function.evaluate(self.x_values_c).unsqueeze(1)
            self.survivors_selection.execute()

            iter_result = self.data_processor.processed_result(self.fitness_function.counter)
            self.final_result_registry.process_iter(iter_result)
            self.register.data_entry([iter_result])

            if t.mean(self.offspring_fitness) >= self.tgt_fitness:
                break

        self.final_result_registry.process_finnish()
        return self.final_result_registry.final_result
