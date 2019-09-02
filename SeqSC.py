import math
from sklearn.cluster import KMeans
import SSVD
import numpy as np
import scipy
import SeqKM
from keras.datasets import mnist
import matplotlib.pyplot as plt

# gussian kernel : should add some number to cordinate
def kernel(xi, uj):
    # uj = uj[0]
    k_ans = 0
    for i in range(0, len(xi)):
        d = (np.absolute(xi[i] - uj[i]))
        x = -1 * d
        k_ans = k_ans + math.exp(x)
    return k_ans


def sum_kernel(xi, anchors, ux):
    r = 0
    for u in ux:
        r = r + kernel(xi, anchors[u])
    return r


def euclidean_distance(a, b):
    return np.sum(np.subtract(a, b) ** 2)


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
    # print(temp)
    return temp


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


def showImage(images, rows, columns):
    fig = plt.figure(figsize=(8, 8))
    for i in range(1, columns * rows + 1):
        img = images[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap='gray')
    plt.show()


def seqsc(x, k, m):
    my_x = transform(x)
    kmeans = KMeans(n_clusters=m,init='k-means++')
    kmeans.fit(my_x)
    anchors = kmeans.cluster_centers_
    # v, label_all, anchors = SeqKM.seqkm(m, my_x, 3 * m)
    pic_anchors = retransform(anchors)
    showImage(pic_anchors, 5, int(m / 5))
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
    for i in range(0, len(x)):
        ux = compute_p_nearest(my_x[i], p, anchors)
        for j in ux:
            z[i][j] = kernel(my_x[i], anchors[j]) / sum_kernel(my_x[i], anchors, ux)
            d[j] = d[j] + z[i][j]
    d = np.diag(d)
    d = scipy.linalg.fractional_matrix_power(d, -0.5)
    for s in range(0, len(x)):
        z_bar[s] = np.matmul(z[s], d)
    A, B, sigma = SSVD.ssvd(z_bar, k)
    A_my = build_A(A, k)
    # v, label_all, anch = SeqKM.seqkm(k, A_my, 3 * k)
    kmeans = KMeans(n_clusters=k)#,init='k-means++')
    kmeans.fit(A_my)
    label_all = kmeans.labels_
    return label_all, label_all, anchors
    # v, label_all, centers = SeqKM.seqkm(k, A_my, m)
    # return label_all, centers, anchors


def show_final_result(X_train, y_train, label_all):

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_train, build_distances_black(X_train), c=label_all, s=20)
    printable = []

    for i in range(0, 11):
        temp = []
        for i in range(0, 11):
            temp.append(0)
        printable.append(temp)

    for i in range(0, len(label_all)):
        printable[y_train[i]][label_all[i]] = printable[y_train[i]][label_all[i]] + 1
        printable[y_train[i]][10]= printable[y_train[i]][10] + 1
        printable[10][label_all[i]]= printable[10][label_all[i]]+1
    printable[10][10] = sum(printable[10])
    result = []
    for i in range(2):
        temp = []
        for i in range(11):
            temp.append(0)
        result.append(temp)
    sigmar = 0
    sigmac = 0
    printable_inverse = np.transpose(printable)
    for i in range(10):
        maxr = max(printable[i][0:10])
        maxc = max(printable_inverse[i][0:10])
        result[0][i] = round(maxr / printable[i][10],2)
        result[1][i] = round(maxc / printable_inverse[i][10],2)
        sigmar = sigmar + maxr
        sigmac = sigmac + maxc
    r0 = sigmar/len(label_all)
    r1 = sigmac/len(label_all)
    result[0][10] = round(r0,2)
    result[1][10] = round(r1,2)

    column = ('c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9','sum_all')
    row = ('T_0','T_1','T_2','T_3','T_4','T_5','T_6','T_7','T_8','T_9','sum_cluster')
    fig, axs = plt.subplots(2,1)
    axs[0].axis('tight')
    axs[0].axis('off')
    axs[1].axis('tight')
    axs[1].axis('off')
    table = axs[0].table(cellText=printable,rowLabels=row,colLabels=column,loc='center')
    row = ['real label','clusters']
    column = ['0','1','2','3','4','5','6','7','8','9','all']
    table2 = axs[1].table(cellText=result, rowLabels=row, colLabels=column, loc='center')

    plt.show()
    return


(X_train, y_train), (X_test, y_test) = mnist.load_data()
k =10
i = 0
X_train = X_train[:50]
y_train = y_train[:50]
z = []
for i in range(10):
    z.append(0)
for y in y_train:
    z[y] = z[y] + 1
print(z)
label_all, centers, anchors = seqsc(X_train, k, 15)
show_final_result(X_train, y_train, label_all )


