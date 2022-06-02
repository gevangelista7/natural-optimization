
import numpy as np
import torch as t
from .MutationMCLWithIsland import MutationMCLWithIsland
from .DetSurvivorSelectionMCLWithMigration import DetSurvivorsSelectionMCLWithMigration
from .DiscreteXUniformSWithIslandConst import DiscreteXUniformSWithIslandConst
from .Register import GARegister, DataPrep, FinalResultProcessor
from .FitnessFunctionWithCounter import FitnessFunctionWithCounter


class EvolutionStrategyWithIslandsConst:
    t.set_grad_enabled(False)

    def __init__(self,
                 individual_dimension,
                 fitness_function: FitnessFunctionWithCounter,
                 tgt_fitness,
                 max_eval,
                 _lambda_island,
                 _mu_island,
                 n_island,
                 migration_period,
                 filename,
                 _tau1=None, _tau2=None, _eps0=None, pop0_dispersion=1, x_lim=(-30, 30)):

        self.device = "cuda" if t.has_cuda else "cpu"

        self.individual_dimension = individual_dimension
        self.tgt_fitness = tgt_fitness
        self.max_eval = max_eval
        self._mu = _mu_island * n_island
        self._lambda = _lambda_island * n_island
        self.migration_period = migration_period

        self._eps0 = 1e-3 if _eps0 is None else _eps0
        self._tau1 = 1 / np.sqrt(2 * individual_dimension) if _tau1 is None else _tau1
        self._tau2 = 1 / np.sqrt(2 * np.sqrt(individual_dimension)) if _tau2 is None else _tau2
        self.x_lim = x_lim

        tags_p = (t.arange(self._mu, device=self.device) % n_island).view(-1, 1)
        self.population = t.concat((tags_p,
                                    t.normal(0, pop0_dispersion, (self._mu, self.individual_dimension * 2),
                                             device=self.device)),
                                   axis=1)
        # self.population_x = self.population[:, 1:self.individual_dimension+1]
        # necessidade de impor diversidade pelas ilhas aqui?

        tags_o = (t.arange(self._lambda, device=self.device) % n_island).view(-1, 1)
        self.offspring = t.concat((tags_o,
                                    t.ones((self._lambda, self.individual_dimension * 2), device=self.device)),
                                    axis=1)

        self.offspring_x = self.offspring[:, 1:self.individual_dimension+1]

        self.recombination = DiscreteXUniformSWithIslandConst(parents_tensor=self.population,
                                                              offspring_tensor=self.offspring,
                                                              device=self.device,
                                                              n_islands=n_island)

        self.mutation = MutationMCLWithIsland(population=self.offspring,
                                              _eps0=self._eps0,
                                              _tau1=self._tau1,
                                              _tau2=self._tau2,
                                              device=self.device,
                                              x_lim=x_lim)

        self.offspring_fitness = t.empty((self._lambda, 1), device=self.device)
        self.fitness_function = fitness_function
        fitness_function.link(evaluation_population=self.offspring_x,
                              fitness_array=self.offspring_fitness)

        # self.sigma_control = SigmaControlIterN(self.mutation)
        # self.sigma_control = SigmaControlRechemberg(mutation=self.mutation, fitness_array=self.offspring_fitness)

        self.survivors_selection = DetSurvivorsSelectionMCLWithMigration(offspring_fitness=self.offspring_fitness,
                                                                         survivors=self.population,
                                                                         offspring=self.offspring,
                                                                         migration_period=migration_period,
                                                                         n_island=n_island)

        self.data_processor = DataPrep(population=self.offspring,
                                       fitness=self.offspring_fitness)

        self.iter_register = GARegister(filename=filename,
                                        algo_name='ES{}Island'.format(n_island),
                                        data_header=['gen_n', 'eval_counter', 'iter_time', 'gen_best_fit',
                                                     'gen_best_idv', 'gen_mean_fit', 'gen_worst_fit'])

        self.final_result_registry = FinalResultProcessor(offspring_fitness=self.offspring_fitness,
                                                          tgt_fitness=self.tgt_fitness)

    def run(self):
        gen_n = 0
        while self.fitness_function.counter < self.max_eval:
            self.recombination.execute()
            self.mutation.execute()
            self.fitness_function.fitness_update()
            self.survivors_selection.execute(gen_n)

            iter_result = self.data_processor.processed_result(gen_n, self.fitness_function.counter)
            self.final_result_registry.process_iter(iter_result)
            self.iter_register.data_entry([iter_result])
            # self.sigma_control.update_sigma(gen_n)

            if t.mean(self.offspring_fitness) >= self.tgt_fitness:
                break
            gen_n += 1

        self.final_result_registry.process_finnish(iter_result)
        return self.final_result_registry.final_result
