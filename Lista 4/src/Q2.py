
from GeneticAlgorithm import EvolutionStrategy, GAEvolutionPlot, GARegister
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
    overall_result = []
    overall_register = GARegister("ES",
                                  data_header=['elapsed_time', 'success', 'max_fit', 'best_idv', 'final_gen_mean_fit',
                                               'final_gen_best_fit', 'final_gen_best_idv'],
                                  filename="res_q2/Q2_150_rodadas")
    for i in range(150):
        ackley_fitness = FitnessFunction(ackley_t_inv)
        result = EvolutionStrategy(individual_dimension=20,
                                   fitness_function=ackley_fitness,
                                   tgt_fitness=-5e-2,
                                   max_ite=1e3,
                                   _eps0=1e-4,
                                   _lambda=500,
                                   _mu=15,
                                   filename="res_q2/rodada{}_".format(i)).run()
        result['final_gen_best_fit'] = result['final_gen_best_fit'].item()
        result['final_gen_mean_fit'] = result['final_gen_mean_fit'].item()
        result['final_gen_best_idv'] = result['final_gen_best_idv'].tolist()
        overall_result.append(result)

    overall_register.data_entry(overall_result)
