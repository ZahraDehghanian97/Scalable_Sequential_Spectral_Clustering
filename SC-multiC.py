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
    for img in X_train :
        temp = []
        for row in img :
            temp.extend(row)
        ans.append(temp)
        # print(temp)
    return ans


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
        printable[y_train[i]][10] = printable[y_train[i]][10] + 1
        printable[10][label_all[i]] = printable[10][label_all[i]] + 1
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
    row = ['real label', 'clusters']
    column = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'all']
    table2 = axs[1].table(cellText=result, rowLabels=row, colLabels=column, loc='center')

    plt.show()
    return



def cd_show_final_result(X_train,y_train,label_all):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_train, build_distances_black(X_train), c=label_all, s=20)
    plt.show()
    printable = []

    for i in range (0 ,10):
        temp = []
        for i in range(0, 10):
            temp.append(0)
        printable.append(temp)
    for i in range (0,len(label_all)):
        printable[label_all[i]][y_train[i]] = printable[label_all[i]][y_train[i]] +1
    true = 0
    allsum = 0
    for i in range (0 ,10):
        true = true + max(printable[i])
        allsum = allsum + sum(printable[i])
        print (str(i)+ " === 0("+str(printable[i][0])+ ") , 1("+str(printable[i][1])+ ") , 2("+str(printable[i][2])+ ") , 3("+str(printable[i][3])+ ") , 4("+str(printable[i][4])+ ") , 5("+str(printable[i][5])+ ") , 6("+str(printable[i][6]) + ") , 7("+str(printable[i][7])+ ") , 8("+str(printable[i][8])+ ") , 9("+str(printable[i][9])+")")
    print ("accuracy : "+ str(true/allsum))
    return

#
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# i = 0
# total_correct = 0
# # to have faster run i slice the samples
# x_train = X_train[:500]
# y_train = y_train[:500]

# A = find_neighbors_graph(x_train)
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



# print(z)
# clustering = SpectralClustering(n_clusters=10,).fit(z)
# show_final_result(x_train,y_train,clustering.labels_)
#


import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition)
import numpy as np
from sklearn import cluster
(X_train, y_train), (X_test, y_test) = mnist.load_data()
x_train = X_train[:500]
y_train = y_train[:500]
z = transform(x_train)
X_spec = manifold.SpectralEmbedding(n_components=2, affinity='nearest_neighbors', gamma=None, random_state=None,
                                    eigen_solver=None, n_neighbors=5).fit_transform(z)
spectral = cluster.SpectralClustering(n_clusters=10, eigen_solver='arpack', affinity="nearest_neighbors")
X = spectral.fit(X_spec)
y_pred = spectral.fit_predict(X_spec)
show_final_result(x_train,y_train,y_pred)

# clustering evaluation metrics
# print(confusion_matrix(y_train, y_pred))
# print(completeness_score(y_train, y_pred))

# with plt.style.context('fivethirtyeight'):
#     plt.title("Spectral embedding & spectral clustering on MNIST")
#     plt.scatter(X_spec[:, 0], X_spec[:, 1], c=y_pred, s=50, cmap=plt.cm.get_cmap("jet", 10))
#     plt.colorbar(ticks=range(10))
#     plt.clim(-0.5, 9.5)
# plt.show()
