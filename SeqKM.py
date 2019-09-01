# import numpy as np
# from keras.datasets import mnist
# import random as rd
# import matplotlib.pyplot as plt
# import math
#
#
# def build_distances_black(x_train):
#     v = []
#     for image in x_train:
#         v.append(np.sum(image * image))
#     return v
#
#
# def find_corner(img):
#     x = 1
#     y = 1
#     flag = True
#     for i in range(0, 27):
#         for j in range(0, 27):
#             if img[i][j] != 0:
#                 y = i
#                 flag = False
#             if not flag:
#                 break
#         if not flag:
#             break
#     flag = True
#     for j in range(0, 27):
#         for i in range(0, 27):
#             if img[i][j] != 0:
#                 x = j
#                 flag = False
#             if not flag:
#                 break
#         if not flag:
#             break
#     return x, y
#
#
# def euclidean_distance(img_a, img_b,k):
#     if k > 10 :
#         count = 0
#         img_a = img_a[0]
#         x1, y1 = find_corner(img_a)
#         x2, y2 = find_corner(img_b)
#         # x1=0
#         # y1=0
#         # x2=0
#         # y2 = 0
#         minx = min(28 - x1, 28 - x2)
#         miny = min(28 - y1, 28 - y2)
#         for i in range(0, minx):
#             for j in range(0, miny):
#                 if not ((img_a[i][j] > 4 and img_b[i][j] > 4) or (img_a[i][j] < 5 and img_b[i][j] < 5)):
#                     count = count + 1
#                 # temp= img_a[i+x1][j+y1] - img_b[i+x2][j+y2]
#                 # count = (temp*temp )+ count
#         return count
#     else :
#         count = 0
#         print("we are here")
#         for i in range(0, len(img_a[0])):
#             temp= img_a[0][i] - img_b[i]
#             count = (temp*temp )+ count
#         return count
#
# def KMeans(images, k):
#     centroids = []
#     probabilities = []
#     i = len(images)
#     while i > 0:
#         probabilities.append(1 / len(images))
#         i = i - 1
#     i = 0
#     print("choose 0's centroid")
#     centroids.append(rd.choices(population=images, weights=probabilities))
#     counter = k - 1
#     while counter > 0:
#         print("choose "+str(k-counter)+"'s centroid")
#         i = 0
#         for image in images:
#             distances = [euclidean_distance(u, image,k) for (u) in centroids]
#             # if len(distances)== 1 and not (type(distances[0]) is int):
#             #     distances = distances[0]
#             probabilities[i] = min(distances)
#             # print(probabilities[i])
#             # print("------------------------------")
#             i = i + 1
#         sumP = sum(probabilities)
#         i = 0
#         for p in probabilities:
#             probabilities[i] = p / sumP
#             i = i + 1
#         centroids.append(rd.choices(population=images, weights=probabilities))
#         counter = counter - 1
#     return centroids
#
#
# def tKMeans(images, k):
#     centroids = []
#     probabilities = []
#     i = len(images)
#     while i > 0:
#         if i == len(images):
#             probabilities.append(1)
#         else:
#             probabilities.append(0)
#         i = i - 1
#     i = 0
#     centroids.append(rd.choices(population=images, weights=probabilities))
#     probabilities[0] = 0
#     probabilities[1] = 1
#     centroids.append(rd.choices(population=images, weights=probabilities))
#     probabilities[1] = 0
#     probabilities[2] = 1
#     centroids.append(rd.choices(population=images, weights=probabilities))
#     probabilities[2] = 0
#     probabilities[3] = 1
#     centroids.append(rd.choices(population=images, weights=probabilities))
#     probabilities[3] = 0
#     probabilities[4] = 1
#     centroids.append(rd.choices(population=images, weights=probabilities))
#     probabilities[4] = 0
#     probabilities[5] = 1
#     centroids.append(rd.choices(population=images, weights=probabilities))
#     probabilities[5] = 0
#     probabilities[7] = 1
#     centroids.append(rd.choices(population=images, weights=probabilities))
#     probabilities[7] = 0
#     probabilities[13] = 1
#     centroids.append(rd.choices(population=images, weights=probabilities))
#     probabilities[13] = 0
#     probabilities[15] = 1
#     centroids.append(rd.choices(population=images, weights=probabilities))
#     probabilities[15] = 0
#     probabilities[17] = 1
#     centroids.append(rd.choices(population=images, weights=probabilities))
#
#     return centroids
#
#
# def showImage(images, rows, columns):
#     fig = plt.figure(figsize=(8, 8))
#     for i in range(1, columns * rows + 1):
#         img = images[i - 1][0]
#         fig.add_subplot(rows, columns, i)
#         plt.imshow(img, cmap='gray')
#     plt.show()
#
#
# def seqkm(k, Images, SampleSize):
#     v = []
#     print(int(k/5))
#     PredictedLabels = []
#     f = k
#     while f > 0:
#         v.append(1)
#         f = f - 1
#     M = rd.choices(Images, k=SampleSize)
#     print("choose "+str(k)+" centroid with kmeans")
#     centers = KMeans(M, k)
#     f = 0
#     if k >10 :
#         showImage(centers, 5, int(k / 5))
#     i = 0
#     for image in Images:
#         distances = [euclidean_distance(centroid, image,k)
#                      for (centroid) in centers]
#         # if len(distances) == 1 and not (type(distances[0]) is int):
#         #     distances = distances[0]
#         print(distances)
#         j = distances.index(min(distances))
#         print(j)
#         PredictedLabels.append(j)
#         i = i + 1
#         v[j] = v[j] + 1
#         epsilon = 1 / v[j]
#         f = f + 1
#         print("update centroid number "+str(j))
#         if k > 10 :
#             for row in range(28):
#                 for col in range(28):
#                     centers[j][0][row][col] = ((1 - epsilon) * centers[j][0][row][col]+0.5) + (epsilon * image[row][col]+0.5)
#         # else :
#         #     for i in range(0,len(image)) :
#         #         centers[j][0][i] = (1- epsilon)*centers[j][0][i] + epsilon*image[i]
#     if k >10 :
#         showImage(centers, 5, int(k / 5))
#     return v, PredictedLabels, centers
#
#
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# i = 0
# total_correct = 0
# X_train = X_train[:200]
# y_train = y_train[:200]
# z = []
# for i in range(10):
#     z.append(0)
# for y in y_train:
#     z[y] = z[y] + 1
# # print(z)
# v, label_all, centers= seqkm(10, X_train,70)
# print(z)
# for i in range(0, 10):
#     z[i] = (v[i] - z[i])
# print(v)
# print(z)
# fig, ax = plt.subplots(figsize=(8, 5))
# ax.scatter(y_train, build_distances_black(X_train), c=label_all, s=20)
# plt.show()
#
# # x_train = [[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]]
# # v, label_all, centers= seqkm(2, x_train,4)
# # print(label_all)