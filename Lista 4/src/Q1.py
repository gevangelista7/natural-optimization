
import torch
from GeneticAlgorithm import StandardGeneticAlgorithm, GAEvolutionPlot
import numpy as np
import pandas as pd




def decode(X):
    return 4*X/2**16 - 2


def y(x):
    x = decode(x)
    return -(x**2 - .3*np.cos(10*np.pi*x))


if __name__ == "__main__":
    torch.no_grad()

    final_result = []
    for i in range(5):
        SGA = StandardGeneticAlgorithm(dimension=16, n_population=30, n_bit_mutate=4, fitness_function=y, max_ite=1e3,
                                       tgt_fitness=0.2999)
        result = SGA.run()
        fig = GAEvolutionPlot(SGA.register.file_name).plot_evolution()
        fig.savefig("Q1_curva{}".format(i+1))

        result['final_gen_best_fit'] = result['final_gen_best_fit'].item()
        result['final_gen_best_idv'] = result['final_gen_best_idv'].item()
        result['final_gen_mean_fit'] = result['final_gen_mean_fit'].item()
        result['fenotipo_best_idv'] = decode(result['best_idv'])
        result['fenotipo_final_gen_best_idv'] = decode(result['final_gen_best_idv'])

        final_result.append(result)

    final_result = pd.DataFrame(final_result)
    print(final_result)





