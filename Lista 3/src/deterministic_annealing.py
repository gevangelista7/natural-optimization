import numpy as np
import numpy.random as rd


def deterministic_annealing(X, n_centroid, T0, Tmin, max_iterations, alpha, epsilon, delta):
    # convection: X -> np.array((number_of_data_points, dimension))
    dim = X.shape[-1]
    Y = rd.normal(0, 1, size=(n_centroid, dim))
    d_xy = np.zeros((len(Y), len(X)))
    p_yx = np.zeros((len(Y), len(X)))

    # initialize histories
    history_J = np.zeros(max_iterations)
    history_J[0] = np.inf

    history_T = np.zeros(max_iterations)
    history_T[0] = T0

    history_D = np.zeros(max_iterations)
    history_D[0] = np.inf

    history_Y = np.zeros((max_iterations, *Y.shape))
    history_Y[0] = Y

    finished = False
    T = T0
    i = 1

    while not finished:
        # Partition condition
        for i_x in range(len(X)):
            for i_y in range(len(Y)):
                d_xy[i_y, i_x] = np.sum((X[i_x] - Y[i_y])**2)

        p_yx = np.exp(-d_xy/T)
        Z_x = np.sum(p_yx, axis=0)
        p_yx = p_yx / Z_x

        # Centroid condition
        for i_y in range(len(Y)):
            Y[i_y] = np.dot(p_yx[i_y], X)/np.sum(p_yx[i_y])

        # Cost Function and history
        history_J[i] = -T/len(X)*np.sum(np.log(Z_x))
        history_D[i] = np.mean(np.sum(p_yx * d_xy, axis=0))
        history_T[i] = T

        # Loop control
        if abs(history_J[i] - history_J[i-1])/abs(history_J[i-1]) < delta:
            T = alpha * T
            Y = Y + epsilon * np.random.normal(0, 1, np.shape(Y))

        history_Y[i] = Y
        i += 1
        if (T < Tmin) or (i == max_iterations):
            finished = True

    # history = {
    #     "Y": history_Y,
    #     "J": history_J,
    #     "D": history_D,
    #     "T": history_T
    # }
    return Y, p_yx, i, history_J, history_D, history_T, history_Y   # ,history


