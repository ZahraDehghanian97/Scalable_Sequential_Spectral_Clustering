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


#
# def affinity(img_a, img_b):
#     count = 0
#     for i in range(0, 28):
#         for j in range(0, 28):
#             if not ((img_a[i][j] > 4 and img_b[i][j] > 4) or (img_a[i][j] < 5 and img_b[i][j] < 5)):
#                 count = count + 1
#     ans = math.exp((count/20))
#     return ans
#     # return count
#
#
# def find_neighbors_graph(x_train):
#     v = []
#     for image1 in x_train:
#         z = []
#         for image2 in x_train:
#             z.append(affinity(image1, image2))
#         v.append(z)
#     return v
#
#
# def make_laplacian(A):
#     b = A
#     i = 0
#     for row in b:
#         z = 0
#         count = 0
#         for cell in row:
#             z = z + cell
#             row[count] = cell * -1
#             count = count + 1
#         row[i] = z
#         i = i + 1
#     return b


def transform(X_train):
    ans = []
    for img in X_train:
        temp = []
        for row in img:
            temp.extend(row)
        ans.append(temp)
        # print(temp)
    return ans


def PlogP(p):
    if p == 0:
        return 0
    log = math.log(p, 2)
    ans = -1 * p
    ans = ans * log
    return ans


def NMI(printable):
    print("compute NMI ------------")
    end_row = len(printable) - 1
    end_col = len(printable[0]) - 1
    H_Y = 0
    H_C = 0
    H_Y_C = 0
    I_Y_C = 0
    for i in range(0, end_row):
        p = printable[i][end_col] / printable[end_row][end_col]
        H_Y = H_Y + PlogP(p)
    for i in range(0, end_col):
        p = printable[end_row][i] / printable[end_row][end_col]
        H_C = H_C + PlogP(p)
    print("H(Y) = " + str(H_Y))
    print("H(C) = " + str(H_C))
    for j in range(0, end_col):
        temp = 0
        print("H (Y | C = " + str(j) + " ) = ", end='')
        P_C = printable[end_row][j] / printable[end_row][end_col]
        for i in range(0, end_row):
            p = printable[i][j] / printable[end_row][j]
            p = P_C * PlogP(p)
            # print("H (Y = " + str(i) + " | C = " + str(j) + " ) = " + str(p) + " +", end=' ')
            temp = temp + p
        print(" ")
        H_Y_C = H_Y_C + temp
    I_Y_C = H_Y - H_Y_C
    ans = (2 * I_Y_C) / (H_Y + H_C)
    ans = round(ans, 3)
    print("H(Y|C) = " + str(H_Y_C))
    print("I(Y;C) = " + str(I_Y_C))
    print("-----------------------")
    print("NMI = " + str(ans))
    return ans


def show_final_result(X_train, y_train, label_all, k):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_train, build_distances_black(X_train), c=label_all, s=20)
    printable = []
    show_centroid(X_train, label_all, k)
    for i in range(0, 11):
        temp = []
        for i in range(0, 11):
            temp.append(0)
        printable.append(temp)

    for i in range(0, len(label_all)):
        printable[y_train[i]][label_all[i]] = printable[y_train[i]][label_all[i]] + 1
        printable[y_train[i]][10] = printable[y_train[i]][10] + 1
        printable[10][label_all[i]] = printable[10][label_all[i]] + 1
    printable[10][10] = sum(printable[10])

    nmi = NMI(printable)

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
        result[0][i] = round(maxr / printable[i][10], 2)
        result[1][i] = round(maxc / printable_inverse[i][10], 2)
        sigmar = sigmar + maxr
        sigmac = sigmac + maxc
    r0 = sigmar / len(label_all)
    r1 = sigmac / len(label_all)
    result[0][10] = round(r0, 2)
    result[1][10] = round(r1, 2)

    column = ('c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'sum_all')
    row = ('T_0', 'T_1', 'T_2', 'T_3', 'T_4', 'T_5', 'T_6', 'T_7', 'T_8', 'T_9', 'sum_cluster')
    fig, axs = plt.subplots(2, 1)
    axs[0].axis('tight')
    axs[0].axis('off')
    axs[1].axis('tight')
    axs[1].axis('off')
    table = axs[0].table(cellText=printable, rowLabels=row, colLabels=column, loc='center')
    row = ['classes', 'clusters']
    column = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'all']
    table2 = axs[1].table(cellText=result, rowLabels=row, colLabels=column, loc='center')
    plt.show()
    return


def show_centroid(X_train, label_all, k):
    centroid = []
    counter = []
    for i in range(0, k):
        counter.append(0)
        print(i)
    print("-----")
    # build black
    black = []
    for i in range(len(X_train[0])):
        temp = []
        for j in range(len(X_train[0][0])):
            temp.append(0)
        black.append(temp)
    for i in range(k):
        centroid.append(black)
    for i in range(0, len(label_all) - 1):
        print("counter " + str(label_all[i]) + " is : " + str(counter[label_all[i]]))
        if counter[label_all[i]] > -1:
            centroid[label_all[i]] = centroid[label_all[i]] + X_train[i]
        else:
            centroid.append(X_train[i])
        counter[label_all[i]] = counter[label_all[i]] + 1
    for s in range(0, k - 1):
        for i in range(0, len(centroid[0]) - 1):
            for j in range(0, len(centroid[0][0]) - 1):
                centroid[s][i][j] = centroid[s][i][j] / counter[s]
    # c0 = []
    # cc0 = 0
    # for i in range(0,len(label_all)):
    #     if label_all[i] == 0 :
    #         c0.append(X_train[i])
    #         cc0 = cc0 +1
    # showImage(c0, int(cc0 / 2), 2)
    showImage(centroid, 2, 5)
    return


def showImage(images, rows, columns):
    fig = plt.figure(figsize=(8, 8))
    for i in range(1, columns * rows + 1):
        img = images[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap='gray')
    # plt.show()

    return


import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition)
import numpy as np
from sklearn import cluster

(X_train, y_train), (X_test, y_test) = mnist.load_data()
x_train = X_train[:700]
y_train = y_train[:700]
z = transform(x_train)
X_spec = manifold.SpectralEmbedding(n_components=2, affinity='nearest_neighbors', gamma=None, random_state=None,
                                    eigen_solver=None, n_neighbors=5).fit_transform(z)
spectral = cluster.SpectralClustering(n_clusters=10, eigen_solver='arpack', affinity="nearest_neighbors")
X = spectral.fit(X_spec)
y_pred = spectral.fit_predict(X_spec)
show_final_result(x_train, y_train, y_pred, 10)

# clustering evaluation metrics
# print(confusion_matrix(y_train, y_pred))
# print(completeness_score(y_train, y_pred))

# with plt.style.context('fivethirtyeight'):
#     plt.title("Spectral embedding & spectral clustering on MNIST")
#     plt.scatter(X_spec[:, 0], X_spec[:, 1], c=y_pred, s=50, cmap=plt.cm.get_cmap("jet", 10))
#     plt.colorbar(ticks=range(10))
#     plt.clim(-0.5, 9.5)
# plt.show()
