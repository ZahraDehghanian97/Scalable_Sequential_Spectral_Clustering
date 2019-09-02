import random as rd


def euclidean_distance(img_a, img_b):
    count = 0
    img_a = img_a[0]
    for i in range(0, len(img_a)):
        temp = img_a[i] - img_b[i]
        count = (temp ** 2) + count
    return count


def KMeansPlusPlus(images, k):
    centroids = []
    probabilities = []
    i = len(images)
    while i > 0:
        probabilities.append(1 / len(images))
        i = i - 1
    i = 0
    print("choose 0's centroid")
    centroids.append(rd.choices(population=images, weights=probabilities)[0])
    counter = k - 1
    while counter > 0:
        print("choose " + str(k - counter) + "'s centroid")
        i = 0
        for image in images:
            distances = [euclidean_distance(u, image) for (u) in centroids]
            probabilities[i] = min(distances)
            # print("------------------------------")
            i = i + 1
        sumP = sum(probabilities)
        i = 0
        for p in probabilities:
            probabilities[i] = p / sumP
            i = i + 1
        centroids.append(rd.choices(population=images, weights=probabilities)[0])
        counter = counter - 1
    return centroids


def seqkm(k, Images, SampleSize):

    v = []
    PredictedLabels = []
    f = k
    while f > 0:
        v.append(1)
        f = f - 1
    M = rd.choices(Images, k=SampleSize)
    print("choose " + str(k) + " centroid with kmeans")
    centers = KMeansPlusPlus(M, k)
    f = 0
    i = 0
    for image in Images:
        distances = [euclidean_distance(centroid, image)
                     for (centroid) in centers]
        j = distances.index(min(distances))
        PredictedLabels.append(j)
        i = i + 1
        v[j] = v[j] + 1
        epsilon = 1 / v[j]
        f = f + 1
        print("update centroid number " + str(j))
        for i in range(0, len(image)):
            centers[j][i] = ((1 - epsilon) * centers[j][i] + 0.5) + (epsilon * image[i] + 0.5)
    return v, PredictedLabels, centers
