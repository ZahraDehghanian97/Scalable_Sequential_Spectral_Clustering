import numpy as np
from sklearn.neighbors import radius_neighbors_graph
from sklearn.neighbors import kneighbors_graph
from keras.datasets import mnist
from collections import defaultdict


def find_majority(labels):
    '''Finds the majority class/label out of the given labels'''
    # defaultdict(type) is to automatically add new keys without throwing error.
    counter = defaultdict(int)
    for label in labels:
        counter[label] += 1

    # Finding the majority class.
    majority_count = max(counter.values())
    for key, value in counter.items():
        if value == majority_count:
            return key


def euclidean_distance(img_a, img_b):
    '''Finds the distance between 2 images: img_a, img_b'''
    # element-wise computations are automatically handled by numpy
    return sum((img_a - img_b) ** 2)

def predict(k, train_images, train_labels,test_image):

    #Predicts the new data-point's category/label by
    #looking at all other training labels
    # distances contains tuples of (distance, label)
    distances = [(euclidean_distance(test_image, image), label)
                    for (image, label) in zip(train_images, train_labels)]
    # sort the distances list by distances
    by_distances = sorted(distances, key=lambda distance: distance[1], reverse=False)
    # extract only k closest labels
    k_labels = [label for (_, label) in by_distances[:k]]
    # return the majority voted label
    return find_majority(k_labels)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
i = 0
total_correct = 0
for test_image in X_test:
    pred = predict(10, X_train, y_train, test_image)
    if pred == y_test[i]:
        total_correct += 1
    acc = (total_correct / (i+1)) * 100
    print('test image['+str(i)+']', '\tpred:', pred, '\torig:', y_test[i], '\tacc:', str(round(acc, 2))+'%')
    i += 1