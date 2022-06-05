import torch as t


class GLAStep:
    def __init__(self, data_vector, original, neighbor, n_clusters):
        self.X = data_vector

        self.n_pop = original.shape[0]
        self.n_y = n_clusters
        self.n_x, self.dim = self.X.shape

        self.original = original.view(self.n_pop, self.n_y, self.dim)
        self.neighbor = neighbor.view(self.n_pop, self.n_y, self.dim)

        self.dx = data_vector.expand(self.n_y, self.n_x, self.dim)
        self.dy = t.empty(self.dx.shape)

        self.X_expanded = self.X.expand(self.n_pop, self.n_x, self.dim)

    def update_neighbors(self):
        # partition condition
        self.dy = self.original.expand(self.n_x, self.n_pop, self.n_y, self.dim).permute(1, 2, 0, 3)
        dists = t.norm(self.dx - self.dy, dim=-1)**2
        YgivenX = t.argmin(dists, dim=-2)

        for cluster_i in range(self.n_y):
            index_x = (YgivenX == cluster_i).nonzero()
            unique_idv = index_x[:, 0].unique()
            for idv_idx in unique_idv:
                self.neighbor[idv_idx, cluster_i, :] = t.mean(self.X_expanded[index_x[:, 0], index_x[:, 1]], dim=0)
