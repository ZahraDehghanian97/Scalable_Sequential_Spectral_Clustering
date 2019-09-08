import random as rd
import math
import Kmeans


def euclidean_distance(img_a, img_b):
    count = 0
    for i in range(0, len(img_a)):
        temp = img_a[i] - img_b[i]
        count = (temp ** 2) + count
    count = math.sqrt(count)
    return count


def seqkm(k, Images, SampleSize):
    print("SeqKM start")
    v = []
    PredictedLabels = []
    f = k
    while f > 0:
        v.append(100)
        f = f - 1
    if SampleSize<len(Images) :
        M = rd.choices(Images, k=SampleSize)
    else :
        M = Images
    print("choose " + str(k) + " centroid with kmeans++")
    centers,label = Kmeans.KMeansPlusplus(M, k)
    f = 0
    i = 0
    # for image in Images:
    #     distances = [euclidean_distance(centroid, image)
    #                  for (centroid) in centers]
    #     j = distances.index(min(distances))
    #     PredictedLabels.append(j)
    #     i = i + 1
    #     v[j] = v[j] + 1
    #     epsilon = 1 / v[j]
    #     f = f + 1
        # if SampleSize< len(Images) :
        #     print("update centroid number " + str(j))
        #     for i in range(0, len(image)):
        #         centers[j][i] = ((1 - epsilon) * centers[j][i] + 0.5) + (epsilon * image[i] + 0.5)
    print("SeqKM done")
    return v, PredictedLabels, centers















from sklearn.cluster import KMeans
def seqKM(k , A):
    print("SeqKM start")
    print("Kmeans++ done")
    kmeans = KMeans(n_clusters=k,init='k-means++')
    kmeans.fit(A)
    print("SeqKM done")
    return kmeans.labels_,kmeans.labels_