
from GeneticAlgorithm import FitnessFunctionWithCounter, BinaryMutationVec, StochasticUniversalSampling, GARegister, \
    DataPrep, GAEvolutionPlot
import torch as t
import numpy as np

from Register import FinalResultProcessor
from UniformCrossoverVec import UniformCrossoverVec


class StandardGeneticAlgorithmVecOriginal:

    def __init__(self, dimension,
                 n_population,
                 n_bit_mutate,
                 pop_rate_mutate,
                 fitness_function: FitnessFunctionWithCounter,
                 max_ite,
                 tgt_fitness,
                 filename,
                 dirname=None):

        self.max_ite = max_ite
        self.tgt_fitness = tgt_fitness

        self.population = t.empty((n_population, dimension), dtype=t.bool, device='cuda')
        self.population.random_()

        self.offspring = t.empty((n_population, dimension), dtype=t.bool, device='cuda')
        self.offspring.random_()

        self.offspring_fitness = t.zeros(n_population)
        self.fitness_function = fitness_function
        self.fitness_function.link(evaluation_population=self.offspring,
                                   fitness_array=self.offspring_fitness)

        self.fitness_function.fitness_update()

        self.mutation = BinaryMutationVec(population=self.population,
                                          pop_rate=pop_rate_mutate,
                                          n_bit=n_bit_mutate)

        self.recombination = UniformCrossoverVec(offspring=self.offspring,
                                                 dimensions=dimension,
                                                 parents=self.population)

        self.survivors_selection = StochasticUniversalSampling(offspring_fitness=self.offspring_fitness,
                                                               selected=self.population,
                                                               candidates=self.offspring)

        self.data_processor = DataPrep(population=self.offspring,
                                       fitness=self.offspring_fitness)

        self.iter_register = GARegister(filename=filename,
                                        dir_name=dirname,
                                        algo_name='ES',
                                        data_header=self.data_processor.header)

        self.final_result_registry = FinalResultProcessor(offspring_fitness=self.offspring_fitness,
                                                          tgt_fitness=self.tgt_fitness,
                                                          seed='rand',
                                                          _lambda=1,
                                                          _mu=1)

    def run(self):
        final_result = {
            'max_fit': -np.inf,
            'best_idv':  None,
            'final_gen_mean_fit': None,
            'final_gen_best_fit': None,
            'final_gen_best_idv': None
        }
        iter_result = self.data_processor.processed_result(self.fitness_function.counter,
                                                           self.fitness_function.counter)
        while self.fitness_function.counter < self.max_ite:
            self.recombination.execute()
            self.mutation.execute()
            self.fitness_function.fitness_update()
            self.survivors_selection.execute()

            iter_result = self.data_processor.processed_result(self.fitness_function.counter,
                                                               self.fitness_function.counter)
            self.iter_register.data_entry([iter_result])
            self.final_result_registry.process_iter(iter_result)

            if iter_result['gen_best_fit'] > final_result['max_fit']:
                final_result['max_fit'] = iter_result['gen_best_fit']
                final_result['best_idv'] = iter_result['gen_best_idv']

            if t.mean(self.offspring_fitness) >= self.tgt_fitness:
                break

        final_result['final_gen_mean_fit'] = t.mean(self.offspring_fitness)
        final_result['final_gen_best_fit'] = t.max(self.offspring_fitness)
        final_result['final_gen_best_idv'] = self.offspring[t.argmax(self.offspring_fitness)]

        self.final_result_registry.process_finnish(iter_result)
        return self.final_result_registry.final_result

class StandardGeneticAlgorithmVecMeme(StandardGeneticAlgorithmVecOriginal):
    def run(self):
        final_result = {
            'max_fit': -np.inf,
            'best_idv': None,
            'final_gen_mean_fit': None,
            'final_gen_best_fit': None,
            'final_gen_best_idv': None
        }
        iter_result = self.data_processor.processed_result(self.fitness_function.counter,
                                                           self.fitness_function.counter)

        while self.fitness_function.counter < self.max_ite:
            self.recombination.execute()
            self.mutation.execute()

            self.fitness_function.fitness_update()

            delta = t.zeros_like(self.offspring_fitness)
            if self.fitness_function.counter % 11 == 0:
                delta = (2*self.offspring_fitness + 3 * t.sin(10*t.pi*self.offspring_fitness) < 0.05)*.25

            self. offspring_fitness += delta
            self.survivors_selection.execute()

            if self.fitness_function.counter % 11 != 0:
                iter_result = self.data_processor.processed_result(self.fitness_function.counter,
                                                                   self.fitness_function.counter)
                self.iter_register.data_entry([iter_result])
                self.final_result_registry.process_iter(iter_result)

                if iter_result['gen_best_fit'] > final_result['max_fit']:
                    final_result['max_fit'] = iter_result['gen_best_fit']
                    final_result['best_idv'] = iter_result['gen_best_idv']
    
                if t.mean(self.offspring_fitness) >= self.tgt_fitness:
                    break

        final_result['final_gen_mean_fit'] = t.mean(self.offspring_fitness)
        final_result['final_gen_best_fit'] = t.max(self.offspring_fitness)
        final_result['final_gen_best_idv'] = self.offspring[t.argmax(self.offspring_fitness)]

        self.final_result_registry.process_finnish(iter_result)
        return self.final_result_registry.final_result

class Q1cost(FitnessFunctionWithCounter):
    def __init__(self, n_dim, n_bit, device):
        super(Q1cost, self).__init__()
        self.n_dim = n_dim
        self.n_bit = n_bit
        self.device = device

        self.min_val = -2
        self.max_val = 2

        self.fitness_array = None
        self.evaluation_population = None

        self.bin_arrays = None
        self.decoded_values = None

        self.decode_mask = 2 ** t.arange(self.n_bit, device=device)

    def link(self, evaluation_population: t.Tensor, fitness_array: t.Tensor):
        super(Q1cost, self).link(evaluation_population=evaluation_population,
                                 fitness_array=fitness_array)
        self.n_pop = evaluation_population.shape[0]
        self.bin_arrays = self.evaluation_population.view(self.n_pop, self.n_dim, self.n_bit)

    def decode_values(self):
        pow2 = self.bin_arrays * self.decode_mask
        pos = t.sum(pow2, dim=-1)
        self.decoded_values = self.min_val + (self.max_val - self.min_val) * pos / 2 ** self.n_bit

    def decode_values_vrf(self, bin_arrays):
        pow2 = bin_arrays * self.decode_mask
        pos = t.sum(pow2, dim=-1)
        return self.min_val + (self.max_val - self.min_val) * pos / 2 ** self.n_bit

    def fitness_update(self):
        super(Q1cost, self).fitness_update()
        self.decode_values()
        self.fitness_array.copy_(-(self.decoded_values**2 - 0.3*t.cos(self.decoded_values)).squeeze())


if __name__ == "__main__":
    n_dim = 1
    n_bit = 16
    n_pop = 20
    device = 'cuda'


    results_registry_meme = GARegister(algo_name="GA_meme",
                                  filename="GA_meme",
                                  dir_name='.',
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

    results_registry_original = GARegister(algo_name="GA_original",
                                  filename="GA_original",
                                  dir_name='.',
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

    for _ in range(100):
        cost_func = Q1cost(n_dim=n_dim, n_bit=n_bit, device=device)

        final_result = StandardGeneticAlgorithmVecOriginal(
            n_population=n_pop,
            n_bit_mutate=4,
            pop_rate_mutate=.5,
            dimension=n_bit,
            tgt_fitness=0.3-1e-2,
            fitness_function=cost_func,
            max_ite=800,
            filename='GA_original'
        ).run()

        results_registry_original.data_entry([final_result])
        cost_func = Q1cost(n_dim=n_dim, n_bit=n_bit, device=device)

        final_result = StandardGeneticAlgorithmVecMeme(
            n_population=n_pop,
            n_bit_mutate=4,
            pop_rate_mutate=.5,
            dimension=n_bit,
            tgt_fitness=0.3 - 1e-2,
            fitness_function=cost_func,
            max_ite=800,
            filename='GA_meme'
        ).run()

        results_registry_meme.data_entry([final_result])

    # plotter = GAEvolutionPlot(GA.iter_register.complete_filename)
    # plotter.title = "GA - Q1"
    # plotter.plot_evolution()
    # print(final_result)


