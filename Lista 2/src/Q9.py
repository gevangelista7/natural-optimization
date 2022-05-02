import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from utils import transition_matrix, invariant_vector


J = {
    1: 4,
    2: 1,
    3: 3,
    4: 2,
    5: 4,
}

if __name__ == "__main__":
        T1 = 1/np.log(2)
        tm_t1 = transition_matrix(T1, J)
        pi_t1 = invariant_vector(tm_t1)
        print("T1 = 1/ln(2)")
        print("M_T1 = \n{}".format(tm_t1))
        print("PI_T1 = {}".format(pi_t1))

        T2 = 1 / np.log(3)
        tm_t2 = transition_matrix(T2, J)
        pi_t2 = invariant_vector(tm_t2)
        print("T2 = 1/ln(3)")
        print("M_T2 = \n{}".format(tm_t2))
        print("PI_T2 = {}".format(pi_t2))
