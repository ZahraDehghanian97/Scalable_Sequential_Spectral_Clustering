import random as rd
import math


def euclidean_distance(img_a, img_b):
    count = 0
    for i in range(0, len(img_a)):
        temp = img_a[i] - img_b[i]
        count = (temp ** 2) + count
    count = math.sqrt(count)
    return count


def KMeansPlusPlus(images, k):
    centroids = []
    probabilities = []
    i = len(images)
    while i > 0:
        probabilities.append(1 / len(images))
        i = i - 1
    i = 0
    print("choose 0's centroid")
    centroids.append(rd.choices(population=images, weights=probabilities)[0])
    counter = k - 1
    while counter > 0:
        print("choose " + str(k - counter) + "'s centroid")
        i = 0
        for image in images:
            distances = [euclidean_distance(u, image) for (u) in centroids]
            probabilities[i] = min(distances)
            # print("------------------------------")
            i = i + 1
        sumP = sum(probabilities)
        i = 0
        for p in probabilities:
            probabilities[i] = p / sumP
            i = i + 1
        centroids.append(rd.choices(population=images, weights=probabilities)[0])
        counter = counter - 1
        print("Kmeans++ done")
    return centroids










# main kmeans for test and compare
from sklearn.cluster import KMeans
def KMeansPlusplus(M, k):
    kmeans = KMeans(n_clusters=k, init='k-means++')
    kmeans.fit(M)
    print("Kmeans++ done")
    return kmeans.cluster_centers_