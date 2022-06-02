class Recombination:
    def __init__(self, offspring_tensor, parents_tensor):
        self.offspring_tensor = offspring_tensor
        self.parents_tensor = parents_tensor
        self.n_parent, self.dimension = parents_tensor.shape
        self.dimension = int((self.dimension/2))

    def execute(self):
        pass

    def parent_selector(self):
        pass
