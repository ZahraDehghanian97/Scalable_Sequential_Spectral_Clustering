from tkinter import *
import numpy as np
import math
import Kmeans
import SC
import matplotlib.pyplot as plt
import SeqSC
from keras.datasets import mnist
from tkinter.filedialog import askopenfilename
import time


def Open():
    askopenfilename(initialdir='Desktop')


def build_distances_black(x_train):
    v = []
    for image in x_train:
        v.append(np.sum(image * image))
    return v


def PlogP(p):
    if p == 0:
        return 0
    log = math.log(p, 2)
    ans = -1 * p
    ans = ans * log
    return ans


def nmi():
    printable = build_printable()
    result = ""
    result = result + "compute NMI ----------------" + "\n\n"
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
    result = result + ("H(Y) = " + str(H_Y)) + "\n\n"
    result = result + ("H(C) = " + str(H_C)) + "\n\n"
    for j in range(0, end_col):
        temp = 0
        # print("H (Y | C = "+str(j)+" ) = ", end = '')
        P_C = printable[end_row][j] / printable[end_row][end_col]
        for i in range(0, end_row):
            p = printable[i][j] / printable[end_row][j]
            p = P_C * PlogP(p)
            # print("H (Y = "+str(i)+" | C = " + str(j) + " ) = "+str(p)+" +", end=' ')
            temp = temp + p
        # print(" ")
        H_Y_C = H_Y_C + temp
    I_Y_C = H_Y - H_Y_C
    ans = (2 * I_Y_C) / (H_Y + H_C)
    ans = round(ans, 3)
    # result = result + ("H(Y|C) = " + str(H_Y_C)) + "\n"
    result = result + ("I(Y;C) = " + str(I_Y_C)) + "\n\n"
    result = result + ("-----------------------") + "\n\n"
    result = result + ("NMI = " + str(ans))
    print("NMI = " + str(ans))
    global asli
    nmiroot = Toplevel(asli)
    nmiroot.title("NMI")
    Label(nmiroot, text=result).grid(row=0, column=0)
    return ans


def centroid():
    global x, labels, k
    centroid = []
    counter = []
    for i in range(0, k.get()):
        counter.append(0)
    # build black
    black = []
    for i in range(len(x[0])):
        temp = []
        for j in range(len(x[0][0])):
            temp.append(0)
        black.append(temp)
    for i in range(k.get()):
        centroid.append(black)
    for i in range(0, len(labels) - 1):
        # print("counter "+str(label_all[i])+" is : "+str(counter[label_all[i]]))
        if counter[labels[i]] > -1:
            centroid[labels[i]] = centroid[labels[i]] + x[i]
        else:
            centroid.append(x[i])
        counter[labels[i]] = counter[labels[i]] + 1
    for s in range(0, k.get() - 1):
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


def showImage(images, rows, columns):
    fig = plt.figure(figsize=(8, 8))
    for i in range(1, columns * rows + 1):
        img = images[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap='gray')
    plt.show()
    return


def build_printable():
    global x, y, labels
    printable = []
    for i in range(0, 11):
        temp = []
        for i in range(0, 11):
            temp.append(0)
        printable.append(temp)

    for i in range(0, len(labels)):
        printable[y[i]][labels[i]] = printable[y[i]][labels[i]] + 1
        printable[y[i]][10] = printable[y[i]][10] + 1
        printable[10][labels[i]] = printable[10][labels[i]] + 1
    printable[10][10] = sum(printable[10])
    return printable


def distribution():
    global labels
    printable = build_printable()
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
    r0 = sigmar / len(labels)
    r1 = sigmac / len(labels)
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


def chart():
    global x, y, labels
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y, build_distances_black(x), c=labels, s=20)
    plt.show()

    return


def show_result():
    root3 = Toplevel(asli)
    root3.title('Results')
    Button(root3, text='NMI', command=nmi, height=2, width=10).grid(padx=3, pady=3, row=0, column=0, sticky=W)
    Button(root3, text='centroids', command=centroid, height=2, width=10).grid(padx=3, pady=3, row=0, column=2,
                                                                               sticky=W)
    Button(root3, text='distribution', command=distribution, height=2, width=10).grid(padx=3, pady=3, row=2, column=0,
                                                                                      sticky=W)
    Button(root3, text='chart', height=2, command=chart, width=10).grid(padx=3, pady=3, row=2, column=2, sticky=W)
    Label(root3, text="""choose report """, width=20).grid(row=1, column=1)
    return


def runseqsc():
    global k, n, m, f, x, y, labels, asli
    t_k = k.get()
    t_n = n.get()
    t_m = m.get()
    t_f = f.get()
    print("k,n,m,f")
    print(t_k, t_n, t_m, t_f)
    if t_n == 0 or t_m == 0 or t_k == 0:
        print("please change 0 value")
        return
    else:
        print("start timer for SeqSC")
        time_start = time.perf_counter()
        x, y, labels = SeqSC.guiseqsc(t_k, t_n, t_m, t_f)
        elapsed = time.perf_counter() - time_start
        print("run time : " + str(elapsed))
        show_result()
    return


def runsc():
    global k, n, f, x, y, labels, asli
    t_k = k.get()
    t_n = n.get()
    t_f = f.get()
    print("k,n,f")
    print(t_k, t_n, t_f)
    if t_n == 0 or t_k == 0:
        print("please change 0 value")
        return
    else:
        print("start timer for SC")
        time_start = time.perf_counter()
        x, y, labels = SC.guisc(t_k, t_n, t_f)
        elapsed = time.perf_counter() - time_start
        print("run time : " + str(elapsed))
        show_result()
    return


def runkmeansplusplus():
    global k, n, f, x, y, labels, asli
    t_k = k.get()
    t_n = n.get()
    t_f = f.get()
    print("k,n,f")
    print(t_k, t_n, t_f)
    if t_n == 0 or t_k == 0:
        print("please change 0 value")
        return
    else:
        print("start timer for KMeans++")
        time_start = time.perf_counter()
        x, y, labels = Kmeans.guikmeansplusplus(t_k, t_n, t_f)
        elapsed = time.perf_counter() - time_start
        print("run time : " + str(elapsed))
        show_result()
    return


def runkmeans():
    global k, n, f, x, y, labels, asli
    t_k = k.get()
    t_n = n.get()
    t_f = f.get()
    print("k,n,f")
    print(t_k, t_n, t_f)
    if t_n == 0 or t_k == 0:
        print("please change 0 value")
        return
    else:
        print("start timer for KMeans")
        time_start = time.perf_counter()
        x, y, labels = Kmeans.guikmeans(t_k, t_n, t_f)
        elapsed = time.perf_counter() - time_start
        print("run time : " + str(elapsed))
        show_result()
    return


def setting():
    global v, k, m, n, f, asli
    type = v.get()
    if type == 1:
        root2 = Toplevel(asli)
        root2.title('Spectral Clustering')
        Label(root2, text="""please check desired value :""").grid(row=0, column=0, columnspan=3)
        Label(root2, text=""" """).grid(row=1, column=0)
        k_entry = Entry(root2, width=7, textvariable=k)
        k_entry.grid(column=1, row=2, sticky=E, padx=5)
        Label(root2, text="k = ").grid(column=0, row=2, sticky=E)
        n_entry = Entry(root2, width=7, textvariable=n)
        n_entry.grid(column=3, row=2, sticky=E, padx=10)
        Label(root2, text="n = ").grid(column=2, row=2, sticky=E)
        Label(root2, text="Current Data Frame :").grid(column=0, row=3, sticky=E)
        Label(root2, text="Mnist    ").grid(column=1, row=3, sticky=E)
        Label(root2, text="New Data (optional):").grid(column=0, row=4, sticky=E)
        Button(root2, text='open', command=Open, height=1).grid(padx=6, pady=3, row=4, column=1, sticky=W)
        Checkbutton(root2, text='filter(0-1)', variable=f).grid(column=3, row=3, columnspan=2, sticky=E)
        Button(root2, text='cluster', command=runsc, height=2, width=15).grid(pady=3, padx=4, row=4, columnspan=3,
                                                                              column=3, sticky=W)
        root2.mainloop()

    if type == 2:
        root2 = Toplevel(asli)
        root2.title('Sequential Spectral Clustering')
        Label(root2, text="""please check desired value :""").grid(row=0, column=0, columnspan=3)
        Label(root2, text=""" """).grid(row=1, column=0)
        k_entry = Entry(root2, width=7, textvariable=k)
        k_entry.grid(column=1, row=2, sticky=E, padx=5)
        Label(root2, text="k = ").grid(column=0, row=2, sticky=E)
        n_entry = Entry(root2, width=7, textvariable=n)
        n_entry.grid(column=3, row=2, sticky=E, padx=10)
        Label(root2, text="n = ").grid(column=2, row=2, sticky=E)
        m_entry = Entry(root2, width=7, textvariable=m)
        m_entry.grid(column=5, row=2, sticky=E, padx=5)
        Label(root2, text="m = ").grid(column=4, row=2, sticky=E)
        Label(root2, text="Current Data Frame :").grid(column=0, row=3, sticky=E)
        Label(root2, text="Mnist    ").grid(column=1, row=3, sticky=E)
        Label(root2, text="New Data (optional):").grid(column=0, row=4, sticky=E)
        Button(root2, text='open', command=Open, height=1).grid(padx=6, pady=3, row=4, column=1, sticky=W)
        Checkbutton(root2, text='filter(0-1)', variable=f).grid(column=3, row=3, columnspan=2, sticky=E)
        Button(root2, text='cluster', command=runseqsc, height=2, width=15).grid(pady=3, padx=4, row=4, columnspan=3,
                                                                                 column=3, sticky=W)
        root2.mainloop()
    if type == 3:
        root2 = Toplevel(asli)
        root2.title('KMeans Clustering')
        Label(root2, text="""please check desired value :""").grid(row=0, column=0, columnspan=3)
        Label(root2, text=""" """).grid(row=1, column=0)
        k_entry = Entry(root2, width=7, textvariable=k)
        k_entry.grid(column=1, row=2, sticky=E, padx=5)
        Label(root2, text="k = ").grid(column=0, row=2, sticky=E)
        n_entry = Entry(root2, width=7, textvariable=n)
        n_entry.grid(column=3, row=2, sticky=E, padx=10)
        Label(root2, text="n = ").grid(column=2, row=2, sticky=E)
        Label(root2, text="Current Data Frame :").grid(column=0, row=3, sticky=E)
        Label(root2, text="Mnist    ").grid(column=1, row=3, sticky=E)
        Label(root2, text="New Data (optional):").grid(column=0, row=4, sticky=E)
        Button(root2, text='open', command=Open, height=1).grid(padx=6, pady=3, row=4, column=1, sticky=W)
        Checkbutton(root2, text='filter(0-1)', variable=f).grid(column=3, row=3, columnspan=2, sticky=E)
        Button(root2, text='cluster', command=runkmeans, height=2, width=15).grid(pady=3, padx=4, row=4, columnspan=3,
                                                                                  column=3, sticky=W)
        root2.mainloop()

    if type == 4:
        root2 = Toplevel(asli)
        root2.title('KMeans++ Clustering')
        Label(root2, text="""please check desired value :""").grid(row=0, column=0, columnspan=3)
        Label(root2, text=""" """).grid(row=1, column=0)
        k_entry = Entry(root2, width=7, textvariable=k)
        k_entry.grid(column=1, row=2, sticky=E, padx=5)
        Label(root2, text="k = ").grid(column=0, row=2, sticky=E)
        n_entry = Entry(root2, width=7, textvariable=n)
        n_entry.grid(column=3, row=2, sticky=E, padx=10)
        Label(root2, text="n = ").grid(column=2, row=2, sticky=E)
        Label(root2, text="Current Data Frame :").grid(column=0, row=3, sticky=E)
        Label(root2, text="Mnist    ").grid(column=1, row=3, sticky=E)
        Label(root2, text="New Data (optional):").grid(column=0, row=4, sticky=E)
        Button(root2, text='open', command=Open, height=1).grid(padx=6, pady=3, row=4, column=1, sticky=W)
        Checkbutton(root2, text='filter(0-1)', variable=f).grid(column=3, row=3, columnspan=2, sticky=E)
        Button(root2, text='cluster', command=runkmeansplusplus, height=2, width=15).grid(pady=3, padx=4, row=4,
                                                                                          columnspan=3,
                                                                                          column=3, sticky=W)
        root2.mainloop()


asli = Tk()
root = Toplevel(asli)
root.title('Clustering')
asli.geometry("5x5")
Label(root, text="""Choose type of clustering :""").grid(row=0, column=0)
Label(root, text=""" """).grid(row=1, column=0)
k = IntVar()
k.set(10)
n = IntVar()
n.set(100)
m = IntVar()
m.set(20)
v = IntVar()
v.set(1)
f = IntVar()
f.set(0)
(x, y), (xt, labels) = mnist.load_data()
Radiobutton(root, text="Spectral Clustering", variable=v, value=1).grid(row=2, column=0)
Radiobutton(root, text="SeqSC", variable=v, value=2).grid(row=2, column=1)
Radiobutton(root, text="Kmeans", variable=v, value=3).grid(row=3, column=0)
Radiobutton(root, text="Kmeans++", variable=v, value=4).grid(row=3, column=1)
Label(root, text=""" """).grid(row=4, column=0)
Button(root, text='cluster', command=setting, height=1, width=10).grid(row=5, columnspan=2, padx=10, column=0, sticky=E)

root.mainloop()
