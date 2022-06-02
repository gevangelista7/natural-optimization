
from GeneticAlgorithm import EvolutionStrategy, GAEvolutionPlot
from OptimTestFunctions import ackley_t_inv
import torch


class FitnessFunction:
    def __init__(self, func):
        self.counter = 0
        self.function = func

    def evaluate(self, X):
        self.counter += 1
        return self.function(X)


if __name__ == "__main__":
    torch.no_grad()
    ackley_fitness = FitnessFunction(ackley_t_inv)
    ES = EvolutionStrategy(individual_dimension=20,
                           fitness_function=ackley_fitness,
                           tgt_fitness=-2e-2,
                           max_ite=1e3,
                           _eps0=1e-4,
                           _lambda=500,
                           _mu=15,
                           filename="teste")

    result = ES.run()

    GAEvolutionPlot(ES.iter_register.file_name).plot_evolution()

    for key in result.keys():
        print(key, " : ", result[key])



