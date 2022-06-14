
import torch as t
from .BinaryMutationVec import BinaryMutationVec
from .UniformCrossoverVec import UniformCrossoverVec
from .StochasticUniversalSampling import StochasticUniversalSampling
from .FitnessFunctionWithCounter import FitnessFunctionWithCounter


class StandardGeneticAlgorithmMultiObj:
    def __init__(self,
                 dimension,
                 n_population,
                 n_bit_mutate,
                 mutation_rate,
                 epoch,
                 n_migrant,
                 fitness_function1: FitnessFunctionWithCounter,
                 fitness_function2: FitnessFunctionWithCounter,
                 max_ite):

        t.no_grad()

        self.epoch = epoch
        self.n_migrant = n_migrant
        self.gen_n = 0

        self.population1 = t.empty((n_population, dimension), dtype=t.bool, device='cuda')
        self.population1.random_()
        self.population2 = t.empty((n_population, dimension), dtype=t.bool, device='cuda')
        self.population2.random_()

        self.offspring1 = t.empty((n_population, dimension), dtype=t.bool, device='cuda')
        self.offspring1.random_()
        self.offspring2 = t.empty((n_population, dimension), dtype=t.bool, device='cuda')
        self.offspring2.random_()

        self.offspring_fitness1 = t.zeros(n_population)
        self.max_ite = max_ite

        self.offspring_fitness2 = t.zeros(n_population)
        self.max_ite = max_ite

        self.fitness_function1 = fitness_function1
        self.fitness_function1.link(evaluation_population=self.offspring1,
                                    fitness_array=self.offspring_fitness1)

        self.fitness_function2 = fitness_function2
        self.fitness_function2.link(evaluation_population=self.offspring2,
                                    fitness_array=self.offspring_fitness2)

        self.mutation1 = BinaryMutationVec(population=self.population1,
                                           pop_rate=mutation_rate,
                                           n_bit=n_bit_mutate, )

        self.mutation2 = BinaryMutationVec(population=self.population2,
                                           pop_rate=mutation_rate,
                                           n_bit=n_bit_mutate, )

        self.recombination1 = UniformCrossoverVec(offspring=self.offspring1,
                                                  dimensions=dimension,
                                                  parents=self.population1)

        self.recombination2 = UniformCrossoverVec(offspring=self.offspring2,
                                                  dimensions=dimension,
                                                  parents=self.population2)

        self.survivors_selection1 = StochasticUniversalSampling(offspring_fitness=self.offspring_fitness1,
                                                                selected=self.population1,
                                                                candidates=self.offspring1)

        self.survivors_selection2 = StochasticUniversalSampling(offspring_fitness=self.offspring_fitness2,
                                                                selected=self.population2,
                                                                candidates=self.offspring2)

    def run(self):

        self.gen_n = 0
        while self.fitness_function1.counter < self.max_ite:
            self.recombination1.execute()
            self.mutation1.execute()
            self.fitness_function1.fitness_update()

            self.recombination2.execute()
            self.mutation2.execute()
            self.fitness_function2.fitness_update()

            if self.gen_n % self.epoch == 0:
                _, best1_idx = self.offspring_fitness1.topk(self.n_migrant)
                _, best2_idx = self.offspring_fitness2.topk(self.n_migrant)

                best1 = t.clone(self.offspring1[best1_idx])

                self.offspring1[best1_idx] = t.clone(self.offspring2[best2_idx])
                self.offspring2[best2_idx] = t.clone(best1)


            self.survivors_selection1.execute()
            self.survivors_selection2.execute()

            self.gen_n += 1

        return t.concat((self.population1, self.population2))

