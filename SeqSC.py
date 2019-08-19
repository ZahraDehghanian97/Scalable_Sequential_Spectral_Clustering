import SSVD
import numpy as np
import scipy
import SeqKM
from keras.datasets import mnist
import matplotlib.pyplot as plt


def kernel(xi, uj):
    return np.sum((xi - uj) * (xi - uj))


def sum_kernel(xi, anchors, ux):
    r = 0
    for u in ux:
        r = r + kernel(xi, anchors[u])
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
    ans = []
    counter = 0
    for anchor in anchors:
        ans.append(counter)
        counter = counter + 1
        distances.append(euclidean_distance(xi, anchor))
    ind = np.argsort(distances)
    # my_anchor = np.array(my_anchor)  # This is the key line
    # ans = my_anchor[ind]
    ans = np.array(ans)
    ans = ans[ind]
    return (ans[0:p])


def normalize(a):
    temp = []
    counter = 0
    for i in a:
        temp.append([])
        for j in i:
            temp[counter].append(j)
        counter = counter + 1
    print(temp)
    return temp


def build_G(A, B, k):
    A = [A[i][0:k] for i in range(0,len(A))]
    B = [B[i][0:k] for i in range(0,len(B))]
    print(B)
    ans = np.concatenate((A, B))
    print(ans)
    return ans


def seqsc(x, k, m):
    v, label_all, anchors = SeqKM.seqkm(m, x, 3 * m)
    d = [0] * m
    z = [[0] * m] * len(x)
    z_bar = [[0] * 1] * len(x)
    p = 15
    for i in range(0, len(x)):
        ux = compute_p_nearest(x[i], p, anchors)
        print(ux)
        for j in ux:
            z[i][j] = kernel(x[i], anchors[j]) / sum_kernel(x[i], anchors, ux)
            d[j] = d[j] + z[i][j]
    d = np.diag(d)
    d = scipy.linalg.fractional_matrix_power(d, -0.5)
    for s in range(0, len(x)):
        z_bar[s] = np.matmul(z[s], d)
    #       z_bar is n*1
    A, B, sigma = SSVD.ssvd(z_bar, k)
    G = build_G(A, B, k)
    v, label_all, centers = SeqKM.seqkm(k, G, m)
    return label_all, centers, anchors



(X_train, y_train), (X_test, y_test) = mnist.load_data()
i = 0
# to have faster run i slice the samples
X_train = X_train[:200]
y_train = y_train[:200]
# 70 is 1/10 of 700 , it is the number of anchor point
z = []
for i in range(10):
    z.append(0)
for y in y_train:
    z[y] = z[y] + 1
print(z)
label_all, centers, anchors = seqsc(X_train, 10, 20)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_train, build_distances_black(X_train), c=label_all[:200], s=20)
plt.show()

