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

    def update_neighbors(self):
        self.dy = self.original.expand(self.n_x, self.n_pop, self.n_y, self.dim).permute(0, 2, 1, 3)
        centroid_list = t.argmin(abs(self.dy - self.dx), axis=-1)







