
import numpy as np
import torch as t
from .MutationMCL import MutationMCL
from .DetSurvivorsSelectionMCL import DetSurvivorsSelectionMCL
from .DiscreteXUniformS import DiscreteXUniformS
from .Register import GARegister, DataPrep, FinalResultProcessor
from .FitnessFunctionWithCounter import FitnessFunctionWithCounter


class EvolutionStrategyWithBoundaryReset:
    t.set_grad_enabled(False)

    def __init__(self,
                 individual_dimension,
                 fitness_function: FitnessFunctionWithCounter,
                 tgt_fitness,
                 max_eval,
                 _lambda,
                 _mu,
                 filename,
                 dirname=None,
                 until_max_eval=False,
                 _tau1=None, _tau2=None, _eps0=None, pop0_dispersion=1, x_lim=(-30, 30)):

        self.device = "cuda" if t.has_cuda else "cpu"

        self.individual_dimension = individual_dimension
        self.tgt_fitness = tgt_fitness
        self.max_eval = max_eval
        self.until_max_eval = until_max_eval
        self._mu = _mu
        self._lambda = _lambda
        self.pop0_dispersion = pop0_dispersion

        self._eps0 = 1e-3 if _eps0 is None else _eps0
        self._tau1 = 1 / np.sqrt(2 * individual_dimension) if _tau1 is None else _tau1
        self._tau2 = 1 / np.sqrt(2 * np.sqrt(individual_dimension)) if _tau2 is None else _tau2
        self.x_lim = x_lim

        self.population = t.normal(0, pop0_dispersion, (self._mu, self.individual_dimension * 2), device=self.device)
        self.population_x = self.population[:, :self.individual_dimension]

        self.offspring = t.ones((self._lambda, self.individual_dimension * 2), device=self.device)
        self.offspring_x = self.offspring[:, :self.individual_dimension]

        self.recombination = DiscreteXUniformS(parents_tensor=self.population,
                                               offspring_tensor=self.offspring,
                                               device=self.device)

        self.mutation = MutationMCL(population=self.offspring,
                                    _eps0=self._eps0,
                                    _tau1=self._tau1,
                                    _tau2=self._tau2,
                                    device=self.device,
                                    x_lim=x_lim)

        self.offspring_fitness = t.empty((_lambda, 1), device=self.device)
        self.fitness_function = fitness_function
        self.fitness_function.link(evaluation_population=self.offspring_x,
                                   fitness_array=self.offspring_fitness)

        self.survivors_selection = DetSurvivorsSelectionMCL(offspring_fitness=self.offspring_fitness,
                                                            survivors=self.population,
                                                            children=self.offspring)

        self.data_processor = DataPrep(population=self.offspring,
                                       fitness=self.offspring_fitness)

        self.iter_register = GARegister(filename=filename,
                                        dir_name=dirname,
                                        algo_name='ES',
                                        data_header=self.data_processor.header)

        self.final_result_registry = FinalResultProcessor(offspring_fitness=self.offspring_fitness,
                                                          tgt_fitness=self.tgt_fitness)

    def reset_bad_individuals(self):
        for idx_idv in range(self.population.shape[0]):
            if (abs(self.population_x[idx_idv]) == self.x_lim[-1]).any():
                self.population_x[idx_idv].normal_(0, self.pop0_dispersion)

    def run(self):
        gen_n = 0
        while self.fitness_function.counter < self.max_eval:
            self.recombination.execute()
            self.mutation.execute()
            self.fitness_function.fitness_update()
            self.survivors_selection.execute()

            iter_result = self.data_processor.processed_result(gen_n, self.fitness_function.counter)
            self.final_result_registry.process_iter(iter_result)
            self.iter_register.data_entry([iter_result])

            if gen_n % 100 == 0:
                self.reset_bad_individuals()

            if t.mean(self.offspring_fitness) >= self.tgt_fitness:
                if not self.until_max_eval:
                    break
            gen_n += 1

        self.final_result_registry.process_finnish(iter_result)
        return self.final_result_registry.final_result