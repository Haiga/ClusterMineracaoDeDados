from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

data0 = load_iris()
print(data0.feature_names)
data = data0.data[:, [2, 3]]

r = 0.50
MinPts = 20
labels = [0] * len(data)
C = 0


def distancia(vetor1, vetor2, r=2):
    tamanho = len(vetor1)
    total = 0
    for i in range(tamanho):
        total += pow(vetor1[i] - vetor2[i], r)
    total = pow(total, 1 / r)
    return total


for P in range(0, len(data)):
    if not (labels[P] == 0):
        continue

    pontos_proximos = []
    for Pn in range(0, len(data)):
        if distancia(data[P], data[Pn]) < r:
            pontos_proximos.append(Pn)

    if len(pontos_proximos) < MinPts:
        labels[P] = -1
    else:
        C += 1
        labels[P] = C
        i = 0
        while i < len(pontos_proximos):
            Pn = pontos_proximos[i]
            if labels[Pn] == -1:
                labels[Pn] = C
            elif labels[Pn] == 0:
                labels[Pn] = C

                pontos_alcance = []
                for Pn in range(0, len(data)):
                    if distancia(data[P], data[Pn]) < r:
                        pontos_alcance.append(Pn)

                if len(pontos_alcance) >= MinPts:
                    pontos_proximos = pontos_proximos + pontos_alcance

            i += 1

cores = ['r', 'g', 'b', 'k', 'y', 'c', 'C0', 'C1', 'C2', 'C3', 'C4']


for i in range(len(labels)):
    labels[i] -= 1
print(labels)
print(data0.target)
# irand = (A+D)/(A+B+C+D)

from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics import jaccard_score

fm_score = fowlkes_mallows_score(data0.target, labels)
jc_score = np.mean(jaccard_score(data0.target, labels, average=None))
print(fm_score)
print(jc_score)
