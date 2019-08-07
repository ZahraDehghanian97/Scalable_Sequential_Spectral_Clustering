import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# (x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = [[[1,0],[0,1]],[[1,0],[0,1]]]


def buildZ(x_train):
    ans = []
    counter = 0
    for x in x_train :
        ans.append([])
        for i in x :
            ans[counter].extend(i)
        counter = counter +1
    return ans


z = buildZ (x_train)
# print(z)
zt = np.transpose(z)
# print(zt)
temp= 0
s = []
v = []
for i in range (0, len(z[0])):
    v.append(0)
for i in range (0 , len(z[0])):
    s.append(v)
for i in range (0 , len(z[0])) :
    for j in range (0 , len(z[0])):
        temp = 0
        for k in range (0, len(z)) :
            temp = temp + zt[i][k]*z[k][j]
        s[i][j] = temp
# print (s)

# calculate zt*z sequentially
