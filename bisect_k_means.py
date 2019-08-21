from sklearn.datasets import load_iris
import random
import matplotlib.pyplot as plt

data0 = load_iris()
# print(data0.feature_names)
data = data0.data[:, [2, 3]]
qtd_centroides = 2

result = random.sample(range(0, len(data)), qtd_centroides)
centroids = []
# print(result)
for i in result:
    centroids.append(data[i])


def distancia(vetor1, vetor2, r=2):
    tamanho = len(vetor1)
    total = 0
    for i in range(tamanho):
        total += pow(vetor1[i] - vetor2[i], r)
    total = pow(total, 1 / r)
    return total


def get_direito(cent_esq, cent):
    cent_dir = [0] * len(cent)
    for i in range(len(cent)):
        cent_dir[i] = (cent_esq[i] - cent[i]) * (-1) + cent[i]
    return cent_dir


iteracao = 0
novo_centroide = []
clusters = []
cluster = []
clusterp = []
centroid = []
# na posicao 0 tem o doscumentos do cluster 0
# na posiccao 1 tem os docs do cluster 1
for cont_iteracao in range(5):
    distancias = []
    for centroid in centroids:
        distancias_ao_centroid = []
        for doc in data:
            distancias_ao_centroid.append(distancia(doc, centroid))
        distancias.append(distancias_ao_centroid)

    cluster = [] * qtd_centroides
    for i in range(qtd_centroides):
        cluster.append([])

    for k in range(len(data)):
        distancias_ao_doc = []
        for i in range(qtd_centroides):
            distancias_ao_doc.append(distancias[i][k])
        num_c = distancias_ao_doc.index(min(distancias_ao_doc))
        cluster[num_c].append(k)

    ###
    x = data[:, 0]
    y = data[:, 1]
    ##
    clusterp = []
    for k in range(len(data)):
        distancias_ao_doc = []
        for i in range(qtd_centroides):
            distancias_ao_doc.append(distancias[i][k])
        clusterp.append(distancias_ao_doc.index(min(distancias_ao_doc)))
    print(cluster)
    ##
    # x = data[:,2]
    # y = data[:,3]
    cores = ['r', 'b', 'c', 'g', 'y', 'k']
    print(centroids)
    for centroid in range(qtd_centroides):
        plt.plot(centroids[centroid][0], centroids[centroid][1], '+' + cores[centroid], linewidth=5, markersize=100)
        for doc in range(len(clusterp)):
            if clusterp[doc] == centroid:
                plt.plot(x[doc], y[doc], '.' + cores[centroid])
    # plt.show()
    ###
    distancias_especificas = [0] * qtd_centroides
    for i in range(qtd_centroides):
        for doc in cluster[i]:
            distancias_especificas[i] += distancias[i][doc]

    # centroid_que_sera_dividido = centroids[distancias_especificas.index(max(distancias_especificas))]
    centroid_que_sera_dividido = distancias_especificas.index(max(distancias_especificas))

    posic = centroid_que_sera_dividido
    sorted_num = random.sample(range(0, len(cluster[posic])), 1)
    centroid_esq = data[cluster[posic][sorted_num[0]]]
    centroid_dir = get_direito(centroid_esq, centroids[posic])

    centroids[posic] = centroid_esq
    centroids.append(centroid_dir)

    qtd_centroides += 1

    # for iterac in range(5):
    #     if iteracao != 0:
    #         centroids = novo_centroide
    #     iteracao += 1
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics import jaccard_score
import numpy as np
fm_score = fowlkes_mallows_score(data0.target, clusterp)
jc_score = np.mean(jaccard_score(data0.target, clusterp, average='micro'))
print(fm_score)
print(jc_score)
