import numpy as np
from keras.datasets import mnist
import random as rd
import matplotlib.pyplot as plt


def build_distances_black(x_train):
    v = []
    for image in x_train:
        v.append(np.sum(image * image))
    return v


def euclidean_distance(img_a, img_b):
    return np.sum((img_a - img_b) * (img_a - img_b))


def find_neighbors_graph(x_train, size):
    v = []
    for image1 in x_train:
        z = []
        for image2 in x_train:
            z.append(euclidean_distance(image1, image2))
        v.append(z)
    return v


def make_laplacian(A):
    i = 0
    for row in A:
        z = 0
        for cell in row:
            z = z + cell
        row[i] = (-1 * z)
        i = i + 1
    return A


(X_train, y_train), (X_test, y_test) = mnist.load_data()
i = 0
total_correct = 0
# to have faster run i slice the samples
x_train = X_train[:70]
y_train = y_train[:70]

A = find_neighbors_graph(x_train, len(x_train))
# print (len(A[69]))

L = make_laplacian(A)
# print ("Laplacian matrix -------------")
# print(L)

eigval, eigvec = np.linalg.eig(L)
# print ("EigenValue matrix -------------")
# print(eigval)

y_spec = []
eigvec = eigvec[:, 10:20]
# print(len(eigvec))

for row in eigvec:
    y_spec.append(np.where(row == (max(row))))
c = []
count = [0,0,0,0,0,0,0,0,0,0]
for y in y_spec:
    c.append(y[0][0])
    count[y[0][0]] = count[y[0][0]] +1

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(y_train, build_distances_black(x_train), c=c, s=25)
plt.show()
print(count)
