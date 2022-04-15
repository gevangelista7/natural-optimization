import copy
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

# estado X = lista ordenada das cardinalidades das cidades visitadas. Começa em zero e termina em zero


def get_aleatory_state(n_cities):
    intermediates = list(range(1, n_cities))
    rd.shuffle(intermediates)
    return [0]+intermediates+[0]


def small_mod_state(X_0):
    X = copy.deepcopy(X_0)
    idx_1 = rd.choice(range(1, len(X) - 1))
    idx_2 = rd.choice(range(1, len(X) - 1))
    X[idx_1], X[idx_2] = X[idx_2], X[idx_1]
    return X


step_cost = {           ## MODIFICAR PARA GENERALIZAR
    0: 0.00,
    1: 1.00,
    2: 1.62,
    3: 1.62,
    4: 1.00
}


def J(X):
    step_distances = [abs(X[i-1] - X[i]) for i in range(len(X))]
    step_costs = [step_cost[d] for d in step_distances]
    return sum(step_costs)


if __name__ == "__main__":
    x_n = get_aleatory_state(5)
    x_min = x_n
    J_n = J(x_n)
    J_min = J_n
    N = 10000
    K = 32
    k = 1

    T_0 = 1
    T = T_0
    epsilon = 5e-3

    n = 0
    finished = False
    history = []

    """Otimização por simulated annealing"""
    while not finished:
        n += 1

        x_possible = small_mod_state(x_n)
        J_possible = J(x_possible)

        deltaJ = J_possible - J_n
        r = rd.uniform(0, 1)

        if r > np.exp(deltaJ / T):
            x_n = x_possible
            J_n = J_possible
            if J_n < J_min:
                J_min = J_n
                x_min = x_n

        history.append((x_min, J_min))
        if n % N == 0:
            k += 1
            T = T_0 / np.log2(1 + k)
            if k > K:
                finished = True

    print("X_min = ", x_min, "\nJ_min = ", J_min)


    """ Geração de estados aleatórios por Algoritmo de Metropolis """

    x_n = get_aleatory_state(5)
    n = 0
    kT = 1
    N = 1000000
    M = 100000
    stable_states = {}

    while n < N:
        possible_x = small_mod_state(x_n)

        deltaJ = J(possible_x) - J(x_n)

        q = np.exp(-deltaJ/kT)
        r = rd.uniform(0, 1)

        a = 0 if r > q else 1

        if deltaJ < 0:
            x_n = possible_x
        else:
            x_n = (1-a) * x_n + a * possible_x

        n += 1
        if n > N - M:
            x_n_tuple = tuple(x_n)
            if x_n_tuple in tuple(stable_states.keys()):
                stable_states[x_n_tuple] += 1
            else:
                stable_states[x_n_tuple] = 1

    states_sorted = dict(sorted(stable_states.items(), key=lambda item: item[1]))
    containers = plt.barh(range(len(states_sorted.keys())), states_sorted.values())
    plt.yticks(range(len(states_sorted.keys())), states_sorted.keys())
    plt.xlabel("Número de ocorrências")
    labels = ["{:.4}".format(i/sum(states_sorted.values())) for i in states_sorted.values()]
    plt.bar_label(containers, labels)

