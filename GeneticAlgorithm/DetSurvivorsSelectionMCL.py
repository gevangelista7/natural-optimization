import torch as t


class DetSurvivorsSelectionMCL:
    def __init__(self, offspring_fitness, survivors, children):
        self.offspring_fitness = offspring_fitness
        self.survivors = survivors
        self.n_survivors = len(survivors)
        self.children = children

    def execute(self):
        expanded = t.concat((self.offspring_fitness, self.children), dim=1)
        expanded = expanded[expanded[:, 0].sort(descending=True)[1]]

        self.survivors[:, :] = expanded[:self.n_survivors, 1:]
