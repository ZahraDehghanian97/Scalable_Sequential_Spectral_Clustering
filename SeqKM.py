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
    # print("choose " + str(k) + " centroid with kmeans++")
    centers,label = Kmeans.KMeansPlusplus(M, k)
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