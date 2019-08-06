import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def build_distances_black(x_train):
    v = []
    for image in x_train:
        v.append(np.sum(image * image))
    return v


def euclidean_distance(img_a, img_b):
    return np.sum((img_a - img_b) * (img_a - img_b))


def find_neighbors_graph(x_train):
    v = []
    for image1 in x_train:
        z = []
        for image2 in x_train:
            z.append(euclidean_distance(image1, image2))
        v.append(z)
    return v


def make_laplacian(A):
    b= A
    i = 0
    for row in b:
        z = 0
        count = 0
        for cell in row:
            z = z + cell
            row[count] = cell * -1
            count = count +1
        row[i] = z
        i = i + 1
    return b


(X_train, y_train), (X_test, y_test) = mnist.load_data()
i = 0
total_correct = 0
# to have faster run i slice the samples
x_train = X_train[:70]
y_train = y_train[:70]

A = find_neighbors_graph(x_train)
# print (len(A[69]))

L = make_laplacian(A)
# print ("Laplacian matrix -------------")
# print(L)

eigval, eigvec = np.linalg.eig(L)
# print ("EigenValue matrix -------------")
x = []

for i, value in enumerate(eigval):
    print("Eigenvector:", eigvec[:, i], ", Eigenvalue:", value)
    x.append(i)

# sort these based on the eigenvalues
eigvec = eigvec[:,np.argsort(eigval)]
eigval = eigval[np.argsort(eigval)]

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(x ,eigval, s=25)
plt.show()
# print(eigval[0:9])

kmeans = KMeans(n_clusters=10).fit(eigvec[:,1:10])
# print(kmeans.labels_)

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(y_train, build_distances_black(x_train), c=kmeans.labels_, s=25)
plt.show()
