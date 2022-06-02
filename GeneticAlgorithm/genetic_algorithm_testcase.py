
import unittest
import torch as t
from MutationMCL import MutationMCL
from MutationMPL import MutationMPL
from DiscreteXUniformS import DiscreteXUniformS
from DiscreteXUniformSWithIslandConst import DiscreteXUniformSWithIslandConst
from DetSurvivorsSelectionMCL import DetSurvivorsSelectionMCL
from OptimTestFunctions import ackley_t_inv
from BinaryMutationInt import BinaryMutationInt
from BinaryMutationVec import BinaryMutationVec
from UniformCrossoverVec import UniformCrossoverVec
from DetSurvivorSelectionMCLWithMigration import DetSurvivorsSelectionMCLWithMigration


t.no_grad()
class FitnessFunction:
    def __init__(self, func):
        self.counter = 0
        self.function = func

    def evaluate(self, X):
        self.counter += 1
        return self.function(X)


class RecombinationTest(unittest.TestCase):
    def test_function(self):
        parents = t.ones((2, 6))
        parents[1] *= 2
        children = t.zeros((1, 6))

        recomb = DiscreteXUniformS(offspring_tensor=children, parents_tensor=parents)
        recomb.execute()

        self.assertFalse((children == 0).any())


class MutationTest(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda'
        self._mu = 30
        self._lambda = 200
        self.dimension = 30

    def test_tau1(self):
        population = t.ones((self._mu, self.dimension), device=self.device)

        mut = MutationMCL(population, _tau1=1, _tau2=0, _eps0=1e-3, rate=None, device=self.device)
        mut.execute()
        print(population)
        self.assertAlmostEqual(0, t.std(population[:, 15:]), delta=0.001)

    def test_tau2_variety(self):
        pop1 = t.ones((self._mu, self.dimension), device=self.device)
        mut = MutationMCL(pop1, _tau1=0, _tau2=1, _eps0=1e-3, rate=None, device=self.device)
        mut.execute()

        self.assertAlmostEqual(1, t.mean(pop1[:, :15]), delta=0.5)

    def test_tau2_min_value(self):
        pop2 = t.ones((self._mu, self.dimension), device=self.device)
        eps0test = 1e-2
        mut2 = MutationMCL(pop2, _tau1=0, _tau2=10, _eps0=eps0test, rate=None, device=self.device)
        mut2.execute()

        self.assertEqual(eps0test, t.min(pop2[:, 15:]))

    def test_mutation_mpl(self):
        pop = t.ones((self._mu, self.dimension), device=self.device)
        for i in range(len(pop)):
            pop[i] *= i
        ofsp = t.zeros((self._lambda, self.dimension), device=self.device)
        mut = MutationMPL(population=pop, offspring=ofsp, _tau1=0, _tau2=1, _eps0=1e-3, rate=None, device=self.device)
        mut.execute()

        print(ofsp)


class SurvivorSelectionTest(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda'
        self._mu = 30
        self._lambda = 200
        self.dimension = 30

    def test_selection(self):
        ofsp = t.ones((3, 6))
        for i in range(len(ofsp)):
            ofsp[i] *= i

        survivors = t.ones((1, 6)) * 10

        fit_func = FitnessFunction(ackley_t_inv)
        ofsp_fit = fit_func.evaluate(ofsp).view(-1,1)

        selector = DetSurvivorsSelectionMCL(ofsp_fit, survivors, ofsp)
        selector.execute()

        self.assertEqual(0, survivors[0][0].item())


class BinaryMutationTest(unittest.TestCase):
    def test_function(self):
        pop = t.arange(16)
        mut = BinaryMutationInt(pop, pop_rate=1 / 16, n_bit=1, dimension=16)
        mut.execute()
        print(pop)


class BinaryMutationVecTest(unittest.TestCase):
    def test_function(self):
        pop = t.concat((t.zeros(1, 10), t.ones(1, 10)), dim=0) > 0
        mut = BinaryMutationVec(population=pop, pop_rate=1, n_bit=5)
        mut.execute()

        print(pop)
        # test por soma 10


class UniformCrossoverVecTest(unittest.TestCase):
    def test_function(self):
        pop = t.concat((t.zeros(1, 10), t.ones(1, 10)), dim=0) > 0
        mut = BinaryMutationVec(population=pop, pop_rate=1, n_bit=5)
        mut.execute()

        print(pop)
        ofsp = t.concat((t.zeros(1, 10), t.zeros(1, 10)), dim=0) > 0
        recomb = UniformCrossoverVec(parents=pop, offspring=ofsp, dimensions=10)
        recomb.execute()
        print(ofsp)


class DiscreteXUniformSWithIslandTest(unittest.TestCase):
    def test_function(self):
        parents = t.ones((4, 6), device='cuda')
        parents[2:] *= 2
        parents[:2, 0] = 0
        parents[2:, 0] = 1

        children = t.zeros((2, 6), device='cuda')
        children[1, 0] = 1

        recomb = DiscreteXUniformSWithIslandConst(offspring_tensor=children,
                                                  parents_tensor=parents,
                                                  n_islands=2,
                                                  device='cuda')
        print(children)
        recomb.execute()
        print(children)

        self.assertFalse((children[1:] == 0).any())


class MigrationTest(unittest.TestCase):
    def test_function(self):
        ofsp = t.ones(5, 11)
        for i in range(5):
            ofsp[i] *= i

        ofsp[0, 0] = 0 #.copy_(t.tensor(0))
        ofsp[1, 0] = 0 # .copy_(t.tensor(0))
        ofsp[2, 0] = 1 #.copy_(t.tensor(2))
        ofsp[3, 0] = 1 #.copy_(t.tensor(2))
        ofsp[4, 0] = 2 # .copy_(t.tensor(3))

        fit_a = t.arange(5).unsqueeze(1) + 10

        survivors = t.zeros(4, 11)

        mig = DetSurvivorsSelectionMCLWithMigration(offspring=ofsp,
                                                    offspring_fitness=fit_a,
                                                    survivors=survivors,
                                                    n_island=3,
                                                    migration_period=1)
        print(survivors)
        mig.execute(1)
        print(survivors)





