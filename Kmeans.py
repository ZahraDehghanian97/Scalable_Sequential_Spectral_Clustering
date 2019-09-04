import random as rd
import math
from keras.datasets import mnist
from sklearn.cluster import KMeans

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



def KMeansPlusplus(M, k):
    kmeans = KMeans(n_clusters=k, init='k-means++')
    kmeans.fit(M)
    print("Kmeans++ done")
    return kmeans.cluster_centers_, kmeans.labels_


def mykmeans(M, k):
    kmeans = KMeans(n_clusters=k, init='random')
    kmeans.fit(M)
    print("Kmeans done")
    return kmeans.cluster_centers_, kmeans.labels_

def transform(X_train):
    ans = []
    for img in X_train:
        temp = []
        for row in img:
            temp.extend(row)
        ans.append(temp)
    return ans


def kmeansplusplus():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    i = 0
    X_train = X_train[:700]
    y_train = y_train[:700]
    k = 10
    my_x = transform(X_train)
    centers, label_all = KMeansPlusplus(my_x, k)
    # show_final_result(X_train, y_train, label_all,k)

def kmeans():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    i = 0
    X_train = X_train[:700]
    y_train = y_train[:700]
    k = 10
    my_x = transform(X_train)
    centers, label_all = mykmeans(my_x, k)
    # show_final_result(X_train, y_train, label_all,k)