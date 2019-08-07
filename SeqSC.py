import SSVD
import numpy as np
import scipy
import SeqKM
from keras.datasets import mnist
import matplotlib.pyplot as plt


def kernel(xi, uj):
    return np.sum((xi - uj) * (xi - uj))


def sum_kernel(xi, u):
    r = 0
    for uj in u:
        r = r + kernel(xi, uj)
    return r


def euclidean_distance(a, b):
    return np.sum(np.subtract(a, b) ** 2)


def build_distances_black(x_train):
    v = []
    for image in x_train:
        v.append(np.sum(image * image))
    return v


def compute_p_nearest(xi, p, anchors):
    distances = []
    my_anchor = anchors
    for anchor in anchors:
        distances.append(euclidean_distance(xi, anchor))
    ind = np.argsort(distances)
    my_anchor = np.array(my_anchor)  # This is the key line
    ans = my_anchor[ind]
    return (ans[0:p])
# problem : i want to take back wich numbers are minimums but it took back the objects

def seqsc(x, k, m):
    label_all, centers, anchors = SeqKM.seqkm(k, x, m)
    d = [0] * m
    z = [[0] * m] * len(x)
    z_bar = [[0] * 1] * len(x)
    p = 15
    for i in range(0, len(x)):
        ux = compute_p_nearest(x[i], p, anchors)
        print(ux)
        for j in ux:
            z[i][j] = kernel(x[i], anchors[j]) / sum_kernel(x[i], anchors[j])
        d = np.sum(d, z[i])
    d = np.diag(d)
    d = scipy.linalg.fractional_matrix_power(d, -0.5)
    for s in range(0, len(x)):
        z_bar[s] = np.matmul(z[s], d)
    #       z_bar is n*1
    A, B, sigma = SSVD.ssvd(z_bar, k)
    label_all, centers, anchors = SeqKM.seqkm(k, A, m)
    return label_all, centers, anchors


(X_train, y_train), (X_test, y_test) = mnist.load_data()
i = 0
# to have faster run i slice the samples
X_train = X_train[:700]
y_train = y_train[:700]
# 70 is 1/10 of 700 , it is the number of anchor point
z = []
for i in range(10):
    z.append(0)
for y in y_train:
    z[y] = z[y] + 1
print(z)
label_all, centers, anchors = seqsc(X_train, 10, 70)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_train, build_distances_black(X_train), c=label_all, s=20)
plt.show()
