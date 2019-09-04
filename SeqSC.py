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


def build_distances_black(x_train):
    v = []
    for image in x_train:
        v.append(np.sum(image * image))
    return v


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
    # plt.show()
    return

def seqsc(x, k, m):
    print("SeqSC start")
    my_x = transform(x)
    v, label_all, anchors = SeqKM.seqkm(m, my_x, 3 * m)
    pic_anchors = retransform(anchors)
    # showImage(pic_anchors, 5, int(m / 5))
    # my_x = transform(x)
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
        for j in ux:
            z[i][j] = kernel(my_x[i], anchors[j]) / sum_kernel(my_x[i], anchors, ux)
            d[j] = d[j] + z[i][j]
    d = np.diag(d)
    d = scipy.linalg.fractional_matrix_power(d, -0.5)
    for s in range(0, len(x)):
        z_bar[s] = np.matmul(z[s], d)
    A, sigma, B = SSVD.ssvd(z_bar)
    A_my = build_A(A, k)
    label_all, centers = SeqKM.seqKM(k, A_my )
    print("SeqSC done")
    return label_all, centers, anchors


def PlogP(p):
    if p == 0 :
        return 0
    log = math.log(p,2)
    ans = -1 * p
    ans = ans * log
    return ans


def NMI(printable):

    print("compute NMI ------------")
    end_row = len(printable) -1
    end_col = len(printable[0]) -1
    H_Y = 0
    H_C = 0
    H_Y_C = 0
    I_Y_C = 0
    for i in range(0,end_row):
        p = printable[i][end_col]/printable[end_row][end_col]
        H_Y = H_Y + PlogP(p)
    for i in range(0,end_col):
        p =printable[end_row][i]/printable[end_row][end_col]
        H_C = H_C + PlogP(p)
    print("H(Y) = "+ str(H_Y))
    print("H(C) = " + str(H_C))
    for j in range(0,end_col):
        temp = 0
        # print("H (Y | C = "+str(j)+" ) = ", end = '')
        P_C = printable[end_row][j]/printable[end_row][end_col]
        for i in range(0,end_row):
            p = printable[i][j]/printable[end_row][j]
            p = P_C * PlogP(p)
            # print("H (Y = "+str(i)+" | C = " + str(j) + " ) = "+str(p)+" +", end=' ')
            temp = temp + p
        # print(" ")
        H_Y_C = H_Y_C + temp
    I_Y_C = H_Y - H_Y_C
    ans = (2 * I_Y_C)/(H_Y+H_C)
    ans = round(ans,3)
    print("H(Y|C) = "+str(H_Y_C))
    print("I(Y;C) = " + str(I_Y_C))
    print("-----------------------")
    print("NMI = "+str(ans))
    return ans


def show_centroid(X_train, label_all,k):
    centroid = []
    counter = []
    for i in range(0,k):
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
    for i in range (0, len(label_all)-1):
        # print("counter "+str(label_all[i])+" is : "+str(counter[label_all[i]]))
        if counter[label_all[i]] >-1:
            centroid[label_all[i]] = centroid[label_all[i]] + X_train[i]
        else :
            centroid.append(X_train[i])
        counter[label_all[i]] = counter[label_all[i]] + 1
    for s in range(0,k-1):
        for i in range(0,len(centroid[0])-1):
            for j in range(0,len(centroid[0][0])-1):
                centroid[s][i][j] = centroid[s][i][j]/counter[s]
    # c0 = []
    # cc0 = 0
    # for i in range(0,len(label_all)):
    #     if label_all[i] == 0 :
    #         c0.append(X_train[i])
    #         cc0 = cc0 +1
    # showImage(c0, int(cc0 / 2), 2)
    showImage(centroid,2,5)




def show_final_result(X_train, y_train, label_all,k):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_train, build_distances_black(X_train), c=label_all, s=20)
    printable = []
    show_centroid(X_train,label_all,k)
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

def make_0_255(X_train):
    for k in range(len(X_train) ):
        for i in range(len(X_train[0])):
            for j in range(len(X_train[0][0])):
                if X_train[k][i][j]<20 :
                    X_train[k][i][j] = 0
                else :
                    X_train[k][i][j] = 255
    return X_train


(X_train, y_train), (X_test, y_test) = mnist.load_data()
i = 0
X_train = X_train[:700]
y_train = y_train[:700]
k = 10
m =int( len(X_train))
z = []
for i in range(10):
    z.append(0)
for y in y_train:
    z[y] = z[y] + 1
# print(z)
X_train = make_0_255(X_train)
label_all, centers, anchors = seqsc(X_train, k, m)
show_final_result(X_train, y_train, label_all,k)
