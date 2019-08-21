from sklearn.datasets import load_iris
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data0 = load_iris()
# print(data0.feature_names)
# data = data0.data[:, [2, 3]]
data = [
    [5, 1, 4, 2],
    [4, 1, 1, 3],
    [6, 6, 6, 3],
    [8, 6, 10, 6],
    [10, 10, 8, 8],
    [9, 8, 7, 4],
]

# result = random.sample(range(0,len(data)), 5)
# centroids = []
# print(result)
cores = ['r', 'g', 'b', 'k', 'y', 'c', 'b', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']


def distancia(vetor1, vetor2, r=2):
    tamanho = len(vetor1)
    total = 0
    for i in range(tamanho):
        total += pow(vetor1[i] - vetor2[i], r)
    total = pow(total, 1 / r)
    # print(total)
    return total


minEx = 2
r = 4

noise = 0
core = 1
border = 2

labels = [0] * len(data)

for i in range(len(data)):
    cont_vizinhos = 0
    for j in range(len(data)):
        d = distancia(data[i], data[j])
        if d <= r:
            cont_vizinhos += 1
        if cont_vizinhos >= minEx:
            break
    if cont_vizinhos >= minEx:
        labels[i] = core
print(labels)
for i in range(len(data)):
    if labels[i] != core:
        for j in range(len(data)):
            if labels[j] == core:
                if distancia(data[i], data[j]) <= r:
                    if i != j:
                        labels[i] = border
#


# for doc in range(len(labels)):
#     if labels[doc] == core:
#         plt.plot(data[doc, 0], data[doc, 1], '.' + cores[core])
#     if labels[doc] == noise:
#         plt.plot(data[doc, 0], data[doc, 1], '.' + cores[noise])
#     if labels[doc] == border:
#         plt.plot(data[doc, 0], data[doc, 1], '.' + cores[border])
# plt.show()

# print(labels.count(1))
# print(labels.count(2))
# print(labels.count(0))
#
# print(labels)

clusters = [-1] * len(data)
cont_cluster = 0
for i in range(len(labels)):
    if labels[i] == core:
        clusters[i] = cont_cluster
        for j in range(len(labels)):
            if clusters[j] == -1:
                if distancia(data[j], data[i]) <= r:
                    clusters[j] = cont_cluster
        cont_cluster += 1

# print(cont_cluster)

n = 100
r = 2 * np.random.rand(n)
theta = 2 * np.pi * np.random.rand(n)
area = 200 * r**2 * np.random.rand(n)
colors = theta

# for doc in range(len(clusters)):
#
#     plt.plot(data[doc, 0], data[doc, 1], '.', colors[clusters[doc]])
plt.xlim(left=0.5)
plt.xlim(right=7)
plt.ylim(top=3)
plt.ylim(bottom=0)
# plt.show()


from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics import jaccard_score

# fm_score = fowlkes_mallows_score(data0.target, clusters)
# jc_score = np.mean(jaccard_score(data0.target, clusters, average='micro'))
# print(fm_score)
# print(jc_score)
print(clusters)
