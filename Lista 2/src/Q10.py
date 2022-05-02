import numpy as np
import matplotlib.pyplot as plt
from utils import transition_matrix, invariant_vector


J1 = .2
# J2 - J1 = -.1*(-2) ->
J2 = .4
# J2 - J3 = -.1*(-1) ->
J3 = .3
# J3 - J4 = -.1*(-2) ->
J4 = .1
# J5 - J4 = -.1*(-1) ->
J5 = .2

J = {
    1: .2,
    2: .4,
    3: .3,
    4: .1,
    5: .2
}


if __name__ == "__main__":
    tm1 = transition_matrix(.1, J)

    e2 = np.exp(-2)
    e1 = np.exp(-1)
    tm2 = np.array([[.5*(1-e2), .5,      .0,            .0,         .5],
                    [.5*e2,     .0,     .5*e1,          .0,         .0],
                    [.0,        .5,     .5*(1 - e1),  .5*e2,        .0],
                    [.0,        .0,      .5,     .5*(2 - e1 - e2),  .5],
                    [.5,        .0,      .0,          .5*e1,        .0]])


    #### item b ####
    PI_M = invariant_vector(tm2)
    print("item b)")
    print("Vetor invariante de probabilidades: \n{}".format(PI_M))


    #### item c ####
    prob_min = min(filter(lambda x: x > 0, tm2.flatten()))

    print("Menor probabilidade -> transição permitida com maior deltaJ.")
    print("max_delta_J = 2")





