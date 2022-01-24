import math
from tensorflow.keras.datasets import (fashion_mnist,mnist)
import matplotlib.pyplot as plt
from sklearn import (manifold, datasets, decomposition)
import numpy as np
from sklearn import cluster
from sklearn.cluster import KMeans


def build_distances_black(x_train):
    v = []
    for image in x_train:
        v.append(np.sum(image * image))
    return v


def transform(X_train):
    ans = []
    for img in X_train:
        temp = []
        for row in img:
            temp.extend(row)
        ans.append(temp)
        # print(temp)
    return ans


def showImage(images, rows, columns):
    fig = plt.figure(figsize=(8, 8))
    for i in range(1, columns * rows + 1):
        img = images[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap='gray')
    # plt.show()

    return


def make_0_255(z):
    X_train = z.copy()
    for k in range(len(X_train)):
        for i in range(len(X_train[0])):
            for j in range(len(X_train[0][0])):
                if X_train[k][i][j] < 20:
                    X_train[k][i][j] = 0
                else:
                    X_train[k][i][j] = 255
    return X_train


def make_0_10(X_train):
    for k in range(len(X_train)):
        for i in range(len(X_train[0])):
            for j in range(len(X_train[0][0])):
                if X_train[k][i][j] < 20:
                    X_train[k][i][j] = 0
                else:
                    X_train[k][i][j] = 10
    return X_train


def kernel(xi, uj):
    # uj = uj[0]
    k_ans = 0
    for i in range(0, len(xi)):
        d = (np.absolute(xi[i] - uj[i]))
        x = d * d
        k_ans = k_ans + x
    k_ans = math.sqrt(k_ans)
    k_ans = math.exp(-1 * k_ans)
    return k_ans


def guisc(k, n, f):
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    x_train = X_train[:n]
    y_train = y_train[:n]
    if (f > 0):
        print("apply filter")
        x_train = make_0_255(x_train)
    z = transform(x_train)
    X_spec = manifold.SpectralEmbedding(n_components=2, affinity='nearest_neighbors', gamma=None, random_state=None,
                                        eigen_solver=None, n_neighbors=5).fit_transform(z)
    spectral = cluster.SpectralClustering(n_clusters=k, eigen_solver='arpack', affinity="nearest_neighbors")
    X = spectral.fit(X_spec)
    y_pred = spectral.fit_predict(X_spec)
    return x_train, y_train, y_pred


def normalize(a):
    print(a)
    ans = []
    for row in a :
        print(row)
        sumz = np.sum(row)
        ans.append( np.divide(row,sumz))
    return ans


def guiSC(k, n, f):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    x_train = X_train[:n]
    y_train = y_train[:n]
    if (f > 0):
        print("apply filter")
        x_train = make_0_10(x_train)
    x_transform = transform(x_train)
    matrix = []
    for x in x_transform:
        temp = []
        for y in x_transform:
            temp.append(-1 * kernel(x, y))
        matrix.append(temp)
    for i in range(len(matrix)):
        matrix[i][i] = -1 * sum(matrix[i])
    # print(matrix[0:10])
    vals, vecs = np.linalg.eig(matrix)
    vecs = vecs[:, np.argsort(vals)]
    # normal_vecs = normalize(vecs[:, len(vecs[0])-k-1:len(vecs[0])-1])
    features = len(vecs[0])
    print(features)
    normal_vecs = vecs[:, 1:k]
    # print(normal_vecs[0])
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(normal_vecs)
    y_pred = kmeans.labels_
    return x_train, y_train, y_pred
