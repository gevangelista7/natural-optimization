import numpy as np
import torch as t


class DetSurvivorsSelectionMCLWithMigration:
    def __init__(self,
                 offspring_fitness,
                 survivors,
                 offspring,
                 n_island,
                 migration_period):
        self.offspring_fitness = offspring_fitness
        self.survivors = survivors
        self.n_survivors = survivors.shape[0]
        self.survivors_dim = int((survivors.shape[1] - 1)/2)
        self.offspring = offspring
        self.migration_period = migration_period
        self.n_island = n_island
        self.randomized_islands = np.arange(self.n_island)

    def execute(self, iter_n):
        expanded = t.concat((self.offspring_fitness, self.offspring), dim=1)
        expanded = expanded[expanded[:, 0].sort(descending=True)[1]]

        if iter_n % self.migration_period == 0:
            np.random.shuffle(self.randomized_islands)
            for island in self.randomized_islands:
                for idx_exp in range(expanded.shape[0]):
                    if expanded[idx_exp, 1] == island:
                        expanded[idx_exp, 1] += 1
                        expanded[idx_exp, 1] %= self.n_island
                        break

        self.survivors[:, :self.survivors_dim+1] = expanded[:self.n_survivors, 1:self.survivors_dim+2]
