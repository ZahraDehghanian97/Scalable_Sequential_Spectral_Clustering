import numpy as np
from keras.datasets import mnist
import random as rd
import matplotlib.pyplot as plt


def build_distances_black(x_train):
    v = []
    for image in x_train:
        v.append(np.sum(image * image))
    return v


def euclidean_distance(img_a, img_b):
    return np.sum((img_a - img_b) * (img_a - img_b))


def KMeans(images, k):
    # choose first one totally random
    centroids = []
    probabilities = []
    i = len(images)
    # intialize with equal probability
    while i > 0:
        probabilities.append(1 / len(images))
        i = i - 1
    i = 0
    centroids.append(rd.choices(population=images, weights=probabilities))
    counter = k - 1
    while counter > 0:
        i = 0
        for image in images:
            distances = [euclidean_distance(centroid, image)
                         for (centroid) in centroids]
            # find index of minimum distance
            probabilities[i] = min(distances)
            print(str(i) + "--" + str(probabilities[i]))
            i = i + 1
        print("------------")
        centroids.append(rd.choices(population=images, weights=(probabilities / sum(probabilities))))
        counter = counter - 1
    return centroids


def seqkm(k, Images, SampleSize):
    # v count the number of image in each cluster
    v = []
    PredictedLabels = []
    f = 10
    # initialize v
    while f > 0:
        v.append(1)
        f = f - 1
    # choose samples to find centroid between those
    M = rd.choices(Images, k=SampleSize)
    centers = KMeans(M, k)
    f = 0
    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = 5
    for i in range(1, columns * rows + 1):
        img = centers[i - 1][0]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap='gray')
    plt.show()
    # find cluster of each image
    for image in Images:
        distances = [euclidean_distance(centroid, image)
                     for (centroid) in centers]
        # find index of minimum distance
        j = distances.index(min(distances))
        PredictedLabels.append(j)
        # add image to that cluster
        v[j] = v[j] + 1
        epsilon = 1 / v[j]
        f = f + 1
        # cordinate centroid with new member
        for row in range(28):
            for col in range(28):
                centers[j][0][row][col] = (1 - epsilon) * centers[j][0][row][col] + epsilon * image[row][col]
    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = 5
    for i in range(1, columns * rows + 1):
        img = centers[i - 1][0]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap='gray')
    plt.show()
    print(str(v))
    return centers , PredictedLabels


(X_train, y_train), (X_test, y_test) = mnist.load_data()
i = 0
total_correct = 0
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
anchors , pred = seqkm(10, X_train, 70)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_train, build_distances_black(X_train), c=pred, s=20)
plt.show()

# validate process
# for i in range(len(X_train)):
#     if pred[i] == y_train[i]:
#         total_correct += 1
#     acc = (total_correct / (i + 1)) * 100
#     print('test image[' + str(i) + ']', '\tpred:', pred[i], '\torig:', y_train[i], '\tacc:', str(round(acc, 2)) + '%')
#     i += 1
