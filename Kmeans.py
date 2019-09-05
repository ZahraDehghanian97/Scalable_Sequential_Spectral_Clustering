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


def kmeans(M, k):
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


def guikmeansplusplus(k, n, f):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train[:n]
    y_train = y_train[:n]
    if (f > 0):
        print("apply filter")
        X_train = make_0_255(X_train)
    my_x = transform(X_train)
    centers, label_all = KMeansPlusplus(my_x, k)
    return X_train, y_train, label_all


def make_0_255(X_train):
    for k in range(len(X_train)):
        for i in range(len(X_train[0])):
            for j in range(len(X_train[0][0])):
                if X_train[k][i][j] < 20:
                    X_train[k][i][j] = 0
                else:
                    X_train[k][i][j] = 255
    return X_train


def guikmeans(k, n, f):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train[:n]
    y_train = y_train[:n]
    if (f > 0):
        print("apply filter")
        X_train = make_0_255(X_train)
    my_x = transform(X_train)
    centers, label_all = kmeans(my_x, k)
    return X_train, y_train, label_all
