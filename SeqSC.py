import math
import SSVD
import numpy as np
import scipy
import SeqKM
from keras.datasets import mnist


# gussian kernel : should add some number to cordinate
def kernel(xi, uj):
    # uj = uj[0]
    k_ans = 0
    for i in range(0, len(xi)):
        d = (np.absolute(xi[i] - uj[i]))
        x = -1 * d
        k_ans = k_ans + math.exp(x)
    return k_ans


def euclidean_distance(a, b):
    return np.sum(np.subtract(a, b) ** 2)


def compute_p_nearest(xi, p, anchors):
    distances = []
    ans = []
    counter = 0
    for anchor in anchors:
        ans.append(counter)
        counter = counter + 1
        distances.append(euclidean_distance(xi, anchor))
    ind = np.argsort(distances)
    ans = np.array(ans)
    ans = ans[ind]
    return (ans[0:p])


def build_A(A, k):
    ans = []
    for block in A:
        ans.append(block[1:k + 1])
    return ans


def transform(X_train):
    ans = []
    for img in X_train:
        temp = []
        for row in img:
            temp.extend(row)
        ans.append(temp)
    return ans


def retransform(X_train):
    ans = []
    for img in X_train:
        temp = []
        for i in range(0, 27):
            s = i * 28
            e = s + 28
            t = []
            # my_img = img[0]
            for i in range(s, e):
                t.append(int(img[i]))
            temp.append(t)
        ans.append(temp)
    return ans


def seqsc(x, k, m):
    print("SeqSC start")
    my_x = transform(x)
    v,label_all, anchors = SeqKM.seqkm(m, my_x,len(my_x))
    p = 5
    d = [0] * m
    lenx = len(x)
    z = []
    for i in range(0, lenx):
        temp = []
        for j in range(0, m):
            temp.append(0)
        z.append(temp)
    z_bar = []
    for i in range(0, lenx):
        z_bar.append([0])
    print("build Z^")
    for i in range(0, len(x)):
        ux = compute_p_nearest(my_x[i], p, anchors)
        sum_k = 0
        for j in ux:
            z[i][j] = kernel(my_x[i], anchors[j])
            sum_k+=z[i][j]
            d[j] = d[j] + z[i][j]
        for j in ux:
            z[i][j]/=sum_k
    d = np.diag(d)
    d = scipy.linalg.fractional_matrix_power(d, -0.5)
    for s in range(0, len(x)):
        z_bar[s] = np.matmul(z[s], d)
    A, sigma, B = SSVD.ssvd(z_bar)
    A_my = build_A(A, k)
    label_all, centers = SeqKM.seqKM(k, A_my)
    print("SeqSC done")
    return label_all, centers, anchors


def make_0_255(X_train):
    for k in range(len(X_train)):
        for i in range(len(X_train[0])):
            for j in range(len(X_train[0][0])):
                if X_train[k][i][j] < 20:
                    X_train[k][i][j] = 0
                else:
                    X_train[k][i][j] = 255
    return X_train


def guiseqsc(k, n, m, f):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train[0:n]
    y_train = y_train[0:n]
    z = []
    for i in range(10):
        z.append(0)
    for y in y_train:
        z[y] = z[y] + 1
    if (f > 0):
        print("apply filter")
        X_train = make_0_255(X_train)
    label_all, centers, anchors = seqsc(X_train, k, m)
    return X_train, y_train, label_all
