import math
from sklearn.cluster import SpectralClustering
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def build_distances_black(x_train):
    v = []
    for image in x_train:
        v.append(np.sum(image * image))
    return v


def affinity(img_a, img_b):
    count = 0
    for i in range(0, 28):
        for j in range(0, 28):
            if not ((img_a[i][j] > 4 and img_b[i][j] > 4) or (img_a[i][j] < 5 and img_b[i][j] < 5)):
                count = count + 1
    ans = math.exp((count/20))
    return ans
    # return count


def find_neighbors_graph(x_train):
    v = []
    for image1 in x_train:
        z = []
        for image2 in x_train:
            z.append(affinity(image1, image2))
        v.append(z)
    return v


def make_laplacian(A):
    b = A
    i = 0
    for row in b:
        z = 0
        count = 0
        for cell in row:
            z = z + cell
            row[count] = cell * -1
            count = count + 1
        row[i] = z
        i = i + 1
    return b


def transform(X_train):
    ans = []
    for img in X_train :
        temp = []
        for row in img :
            temp.extend(row)
        ans.append(temp)
        print(temp)
    return ans




(X_train, y_train), (X_test, y_test) = mnist.load_data()
i = 0
total_correct = 0
# to have faster run i slice the samples
x_train = X_train[:700]
y_train = y_train[:700]

A = find_neighbors_graph(x_train)
# # print (len(A[69]))
#
# L = make_laplacian(A)
# # print ("Laplacian matrix -------------")
# # print(L)
#
# eigval, eigvec = np.linalg.eig(L)
# # print ("EigenValue matrix -------------")
# x = []
#
# for i, value in enumerate(eigval):
#     print("Eigenvector:", eigvec[:, i], ", Eigenvalue:", value)
#     x.append(i)
#
# # sort these based on the eigenvalues
# eigvec = eigvec[:, np.argsort(eigval)]
# eigval = eigval[np.argsort(eigval)]
#
# fig, ax = plt.subplots(figsize=(6, 4))
# ax.scatter(x, eigval, s=25)
#
# # print(eigval[0:9])
#
# kmeans = KMeans(n_clusters=10).fit(eigvec[:, 1:10])
# # print(kmeans.labels_)
#
# fig, ax = plt.subplots(figsize=(6, 4))
# ax.scatter(y_train, build_distances_black(x_train), c=y_train, s=25)
#
#
# fig, ax = plt.subplots(figsize=(6, 4))
# ax.scatter(y_train, build_distances_black(x_train), c=kmeans.labels_, s=25)
# plt.show()


z = transform(x_train)
# print(z)
# clustering = SpectralClustering(n_clusters=10,).fit(z)
# fig, ax = plt.subplots(figsize=(6, 4))
# ax.scatter(y_train, build_distances_black(x_train), c=clustering.labels_, s=25)
# plt.show()



from sklearn.metrics.cluster import completeness_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition)
import numpy as np
import time
from sklearn import cluster
from sklearn.neighbors import kneighbors_graph
from sklearn.utils import shuffle
from matplotlib import offsetbox


# Spectral embedding projection
print("Computing Spectral embedding")
start = int(round(time.time() * 1000))
X_spec = manifold.SpectralEmbedding(n_components=2, affinity='nearest_neighbors', gamma=None, random_state=None,
                                    eigen_solver=None, n_neighbors=5).fit_transform(z)
end = int(round(time.time() * 1000))
print("--Spectral Embedding finished in ", (end - start), "ms--------------")
print("Done.")


# spectral clustering, fitting and predictions
spectral = cluster.SpectralClustering(n_clusters=10, eigen_solver='arpack', affinity="nearest_neighbors")

# X = spectral.fit(X_iso)
X = spectral.fit(X_spec)

# y_pred = spectral.fit_predict(X_iso)
y_pred = spectral.fit_predict(X_spec)
fig, ax = plt.subplots(figsize=(8, 5))
print(len(y_train))
print(len(x_train))
print(len(y_pred))
ax.scatter(y_train, build_distances_black(x_train), c=y_pred, s=20)
plt.show()

# clustering evaluation metrics
print(confusion_matrix(y_train, y_pred))
print(completeness_score(y_train, y_pred))

with plt.style.context('fivethirtyeight'):
    plt.title("Spectral embedding & spectral clustering on MNIST")
    plt.scatter(X_spec[:, 0], X_spec[:, 1], c=y_pred, s=50, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
plt.show()
