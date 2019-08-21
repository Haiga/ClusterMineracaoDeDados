from sklearn.datasets import load_iris
import random
import matplotlib.pyplot as plt

data0 = load_iris()
print(data0.feature_names)
data = data0.data[:, [2, 3]]

# data = [
#     [5, 1, 4, 2],
#     [4, 1, 1, 3],
#     [6, 6, 6, 3],
#     [8, 6, 10, 6],
#     [10, 10, 8, 8],
#     [9, 8, 7, 4],
# ]

numk = 3

result = random.sample(range(0, len(data)), numk)
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


iteracao = 0
novo_centroide = []
cluster = []
for iterac in range(10):
    if iteracao != 0:
        centroids = novo_centroide
    iteracao += 1

    distancias = []
    for centroid in centroids:
        distancias_ao_centroid = []
        for doc in data:
            distancias_ao_centroid.append(distancia(doc, centroid))
        distancias.append(distancias_ao_centroid)

    # print(distancias[1])
    cluster = []
    for k in range(len(data)):
        distancias_ao_doc = []
        for i in range(numk):
            distancias_ao_doc.append(distancias[i][k])
        cluster.append(distancias_ao_doc.index(min(distancias_ao_doc)))
    print(cluster)

    x = data[:, 0]
    y = data[:, 1]
    # x = data[:,2]
    # y = data[:,3]
    cores = ['r', 'b', 'c', 'g', 'y', 'k']
    print(centroids)
    for centroid in range(numk):
        plt.plot(centroids[centroid][0], centroids[centroid][1], '+' + cores[centroid], linewidth=5, markersize=100)
        for doc in range(len(cluster)):
            if cluster[doc] == centroid:
                plt.plot(x[doc], y[doc], '.' + cores[centroid])
            plt.show()

    novo_centroide = centroids
    for i in range(numk):
        for feature in range(len(data[0])):
            total = 0
            cont = 0
            for doc in range(len(data)):
                if cluster[doc] == i:
                    cont += 1
                    total += data[doc][feature]
            total = total / cont
            novo_centroide[i][feature] = total

    # print(novo_centroide)

plt.show()
# print(data0.target)
print(cluster)
