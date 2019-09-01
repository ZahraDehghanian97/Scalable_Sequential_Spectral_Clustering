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
    k_ans = 0
    for i in range (0 , len(xi)):
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
    # print(temp)
    return temp


def build_G(A, B, k):
    A = [A[i][0:k] for i in range(0,len(A))]
    B = [B[i][0:k] for i in range(0,len(B))]
    print(B)
    ans = np.concatenate((A, B))
    print(ans)
    return ans


def build_A(A, k):
    ans = []
    for block in A :
        ans.append(block[1:k+1])
        # ans.append(block[len(block)-k : len(block)])
    # print(ans)
    return ans


def transform(X_train):
    ans = []
    for img in X_train :
        temp = []
        for row in img :
            temp.extend(row)
        ans.append(temp)
    return ans
def retransform(X_train):
    ans = []
    for img in X_train :
        temp = []
        for i in range(0,27) :
            s = i * 28
            e = s+28
            temp.append(img[s:e])
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
    kmeans = KMeans(n_clusters=m)
    kmeans.fit(my_x)
    anchors = kmeans.cluster_centers_
    pic_anchors = retransform (anchors)
    showImage(pic_anchors, 5, int(m / 5))
    # v, label_all, anchors = SeqKM.seqkm(m, x, 3 * m)
    p = 5
    d = [0] * m
    lenx = len(x)
    z = []
    for i in range (0,lenx):
        temp = []
        for j in range(0,m):
            temp.append(0)
        z.append(temp)
    # z_bar = [[0] * 1] * len(x)
    z_bar = []
    for i in range (0,lenx):
        z_bar.append([0])
    for i in range(0, len(x)):
        ux = compute_p_nearest(my_x[i], p, anchors)
        for j in ux:
            z[i][j] = kernel(my_x[i], anchors[j]) / sum_kernel(my_x[i], anchors, ux)
            d[j] = d[j] + z[i][j]
    # print("z is : --------------")
    # print(z)
    # print("d is : ----------------")
    # print(d)
    d = np.diag(d)
    # print("d after diag is : ----------------")
    # print(d)
    d = scipy.linalg.fractional_matrix_power(d, -0.5)
    for s in range(0, len(x)):
        z_bar[s] = np.matmul(z[s], d)
    # print("z bar is  : ----------------")
    # print(z_bar)
    A, B, sigma = SSVD.ssvd(z_bar, k)
    # print("z bar dimansion :" + str(len(z_bar)) +" , "+ str(len(z_bar[0])))
    # print("sigma is : ----------")
    # print(sigma)
    # # print("sigma dimansion :" + str(len(sigma)) )
    # print("A is : ----------")
    # print(A)
    # print("A dimansion :" + str(len(A))  +" , "+str(len(A[0])))
    # # print("B is : ----------")
    # # print(B)
    # print("B dimansion :" + str(len(B)) +" , "+ str(len(B[0])))
    A_my = build_A(A, k)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(A_my)
    label_all = kmeans.labels_
    return label_all,label_all,anchors
    # v, label_all, centers = SeqKM.seqkm(k, A_my, m)
    # return label_all, centers, anchors


def show_final_result(X_train,y_train,label_all):
    # check = []
    # true = 0
    # print(label_all)
    # check.append(label_all[1])
    # check.append(label_all[3])
    # check.append(label_all[5])
    # check.append(label_all[7])
    # check.append(label_all[2])
    # check.append(label_all[0])
    # check.append(label_all[13])
    # check.append(label_all[15])
    # check.append(label_all[17])
    # check.append(label_all[4])
    # for y in y_train :
    #     if y == check[y] :
    #         true = true +1
    # print ("true rate : "+ str(true/len(y_train)))
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
        print(printable[label_all[i]][y_train[i]])
        printable[label_all[i]][y_train[i]] = printable[label_all[i]][y_train[i]] +1
    true = 0
    allsum = 0
    for i in range (0 ,10):
        true = true + max(printable[i])
        allsum = allsum + sum(printable[i])
        print (str(i)+ " === 0("+str(printable[i][0])+ ") , 1("+str(printable[i][1])+ ") , 2("+str(printable[i][2])+ ") , 3("+str(printable[i][3])+ ") , 4("+str(printable[i][4])+ ") , 5("+str(printable[i][5])+ ") , 6("+str(printable[i][6]) + ") , 7("+str(printable[i][7])+ ") , 8("+str(printable[i][8])+ ") , 9("+str(printable[i][9])+")")
    print ("accuracy : "+ str(true/allsum))
    return


(X_train, y_train), (X_test, y_test) = mnist.load_data()
i = 0
X_train = X_train[:700]
y_train = y_train[:700]
z = []
for i in range(10):
    z.append(0)
for y in y_train:
    z[y] = z[y] + 1

label_all, centers, anchors = seqsc(X_train, 10, 70)
show_final_result(X_train,y_train,label_all)


