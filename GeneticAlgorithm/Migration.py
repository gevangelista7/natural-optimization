import torch as t


class Migration:
    def __init__(self, offspring, offspring_fitness, n_island):
        self.offspring = offspring
        self.offspring_fitness = offspring_fitness
        self.n_island = n_island

    def execute(self):
        expanded = t.concat((self.offspring_fitness, self.offspring), dim=1)
        self.offspring.copy_(expanded[expanded[:, 0].sort(descending=True)[1]])

        print(self.offspring)
        for island in range(self.n_island):
            for i in range(len(self.offspring)):
                if self.offspring[i, 1] == island:
                    self.offspring[i, 1] += 1
                    self.offspring[i, 1] %= self.n_island
                    break
        self.offspring = self.offspring[:, 1:]
