import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.spatial import Voronoi, voronoi_plot_2d

# Data

np.random.seed(1);
P = 100;
NC = 9;
# cluster_centers = np.random.normal(0, 1, [2, NC])
#
# aux = np.zeros([2, 1]);
# aux[0] = cluster_centers[0, 0];
# aux[1] = cluster_centers[1, 0]
# data_vectors = 0.1 * np.random.normal(0, 1, [2, P]) + np.tile(aux, (1, P))
# for k in range(1, NC):
#     aux = np.zeros([2, 1]);
#     aux[0] = cluster_centers[0, k];
#     aux[1] = cluster_centers[1, k]
#     data_vectors = np.concatenate((data_vectors, 0.1 * np.random.normal(0, 1, [2, P]) + np.tile(aux, (1, P))), axis=1)

# Main Loop

cluster_centers = np.random.normal(0, 1, [NC, 2])
data_vectors = []
for center in cluster_centers:
    data_vectors.append(center)
    for _ in range(P-1):
        data_vectors.append(center + np.random.normal(0, .1, 2))
data_vectors = np.array(data_vectors).T
X = data_vectors
M, N = np.shape(X);
K = NC;
Y = np.random.normal(0, 1, [M, K]);
T = 10;
alpha = 0.9;
i = 0;
fim = 0;
epsilon = 1e-6;
delta = 1e-3;
d = np.zeros([K, N])
p_ygivenx = np.zeros([K, N])
I = 200
J = np.zeros(I)
D = np.zeros(I)
LocalT = np.zeros(I)

while not (fim):

    # Partition Condition
    for n in range(0, N):
        for k in range(0, K):
            d[k, n] = np.sum(np.power(X[:, n] - Y[:, k], 2))
            p_ygivenx[k, n] = np.exp(-d[k, n] / T)
    Zx = np.sum(p_ygivenx, axis=0)
    p_ygivenx = p_ygivenx / np.tile(Zx, (K, 1))

    # Centroid Condition
    Y = np.zeros([M, K])
    for k in range(0, K):
        y = np.zeros(M)
        w = 0
        for n in range(0, N):
            y += p_ygivenx[k, n] * X[:, n]
            w += p_ygivenx[k, n]
        Y[:, k] = y / w

    # Cost Function and Loop Control
    J[i] = -T / N * np.sum(np.log(Zx))
    D[i] = np.mean(np.sum(p_ygivenx * d, axis=0))
    LocalT[i] = T
    if i == 34: Y34 = Y  # A few codebook examples at critical temperatures
    if i == 45: Y45 = Y
    if i == 67: Y67 = Y
    if i == 90: Y90 = Y
    if (i > 0):
        if abs(J[i] - J[i - 1]) / abs(J[i - 1]) < delta:
            T = alpha * T
            Y = Y + epsilon * np.random.normal(0, 1, np.shape(Y))
    # print([i, J[i], D[i], LocalT[i]])
    i += 1
    if (T < 0.1) or (i == I): fim = 1

plt.rc('font', size=16, weight='bold')

plt.figure()
plt.plot(-J, 'r-', D, 'k-', LocalT, 'b-')
plt.ylim(0, 5)
plt.grid()
#
plt.figure()
plt.plot(-J, 'r-', D, 'k-', LocalT, 'b-')
plt.ylim(0, 3)
plt.xlim(20, 185)
plt.grid()
#
# plt.figure()
# plt.plot(-J, 'r.-', D, 'k.-', LocalT, 'b.-')
# plt.ylim(0.2, 3)
# plt.xlim(20, 100)
# plt.grid()
#
# vor = Voronoi(Y.T)
# fig = voronoi_plot_2d(vor)
# plt.plot(data_vectors[0, :], data_vectors[1, :], 'k.')
# plt.plot(Y34[0, :], Y34[1, :], 'r.', markersize=20)
# plt.xlim(-3, 3)
# plt.ylim(-3, 3)
# plt.grid()
# plt.show()
#
# vor = Voronoi(Y.T)
# fig = voronoi_plot_2d(vor)
# plt.plot(data_vectors[0, :], data_vectors[1, :], 'k.')
# plt.plot(Y45[0, :], Y45[1, :], 'r.', markersize=20)
# plt.xlim(-3, 3)
# plt.ylim(-3, 3)
# plt.grid()
# plt.show()
#
# vor = Voronoi(Y.T)
# fig = voronoi_plot_2d(vor)
# plt.plot(data_vectors[0, :], data_vectors[1, :], 'k.')
# plt.plot(Y67[0, :], Y67[1, :], 'r.', markersize=20)
# plt.xlim(-3, 3)
# plt.ylim(-3, 3)
# plt.grid()
# plt.show()
#
# vor = Voronoi(Y.T)
# fig = voronoi_plot_2d(vor)
# plt.plot(data_vectors[0, :], data_vectors[1, :], 'k.')
# plt.plot(Y90[0, :], Y90[1, :], 'r.', markersize=20)
# plt.xlim(-3, 3)
# plt.ylim(-3, 3)
# plt.grid()
# plt.show()

vor = Voronoi(Y.T)
fig = voronoi_plot_2d(vor)
plt.plot(data_vectors[0, :], data_vectors[1, :], 'k.')
plt.plot(Y[0, :], Y[1, :], 'r.', markersize=20)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.grid()
plt.show()

