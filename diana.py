from sklearn.datasets import load_iris
import numpy as np
import math
import random

from collections import Counter
import matplotlib.pyplot as plt

data0 = load_iris()
# print(data0.feature_names)
data = data0.data[:, [2, 3]]

def distancia(vetor1, vetor2, r=2):
    tamanho = len(vetor1)
    total = 0
    for i in range(tamanho):
        total += pow(vetor1[i] - vetor2[i], r)
    total = pow(total, 1 / r)
    return total


def dispersao(grupo1, grupo2):
    matriz_distancias = []
    for item1 in grupo1:
        vetor_distancias = []
        for item2 in grupo2:
            vetor_distancias.append(distancia(item1, item2))
        matriz_distancias.append(vetor_distancias)
    return matriz_distancias

def min_distance(grupo1, grupo2):
    min_distances = []
    for item1 in grupo1:
        vetor_distancias = []
        for item2 in grupo2:
            vetor_distancias.append(distancia(item1, item2))
        min_distances.append(min(vetor_distancias))
    return min(min_distances)


qtd_documentos = len(data)
grupos = [0]*qtd_documentos
for i in range(qtd_documentos):
    grupos[i] = [i]

def append(id, id2):
    for i in range(qtd_documentos):
        if(grupos[i] == id):
            grupos[i] = id2
mins = [0]*qtd_documentos
matrix = [[0 for x in range(qtd_documentos)] for y in range(qtd_documentos)]
for i in range(qtd_documentos):
    for k in range(qtd_documentos):
        if (i == k):
            matrix[i][k] = math.inf
        else:
            matrix[i][k] = distancia(data[i], data[k])





t = qtd_documentos
max = 5
while t!= max:
    min_i = math.inf
    min_k = math.inf
    mim = math.inf
    for i in range(qtd_documentos):
        for k in range(qtd_documentos):
            if mim > matrix[i][k]:
                    mim = matrix[i][k]
                    min_i = i
                    min_k = k
    matrix[min_i][min_k] = math.inf
    matrix[min_k][min_i] = math.inf
    print(mim)
    print(min_i)
    print(min_k)
    id_group = grupos[min_i]
    id_group2 = grupos[min_k]
    append(id_group,id_group2)
    values, counts = np.unique(grupos, return_counts=True)
    print(len(values))
    t = len(values)
print(grupos)
n = 150
teste =[]
t = 1
cores = ['r', 'g', 'b', 'k', 'y', 'c', 'b', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']

for doc in range(len(grupos)):
    if(grupos[doc] not in teste):
        teste.append(grupos[doc])
    t = teste.index(grupos[doc])
    plt.plot(data[doc, 0], data[doc, 1], '.'+cores[t])


plt.show()















