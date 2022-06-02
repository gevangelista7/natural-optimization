import torch as t
import numpy.random as rd


class StochasticUniversalSampling:
    def __init__(self, offspring_fitness, selected, candidates):
        t.no_grad()
        self.offspring_fitness = offspring_fitness
        self.selected = selected
        self.n_selected = len(selected)
        self.candidates = candidates

    def execute(self):
        adj_ofsp_fit = self.offspring_fitness - t.min(self.offspring_fitness)
        if t.sum(adj_ofsp_fit) == 0:
            prob = t.ones(size=adj_ofsp_fit.shape)
            prob /= len(prob)
        else:
            prob = adj_ofsp_fit / t.sum(adj_ofsp_fit)
        cum_prob = t.cumsum(prob, dim=0)
        r = rd.uniform(0, 1 / self.n_selected)
        id_sel = 0
        i = 0
        while id_sel < self.n_selected:
            if i < len(cum_prob):
                while r < cum_prob[i]:
                    self.selected[id_sel] = self.candidates[i]
                    r += 1 / self.n_selected
                    id_sel += 1
            else:
                self.selected[id_sel] = self.selected[-1]
                id_sel += 1
            i += 1
