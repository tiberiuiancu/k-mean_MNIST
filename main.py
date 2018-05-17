import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# MNIST data parsing
data = input_data.read_data_sets("MNIST_data/", one_hot=False)
clustering_data = data.train.images
testing_data = data.test


def euclidian_distance(image1, image2):
    return np.linalg.norm(image1 - image2)


def reposition_centroids():
    global centroids
    accumulated = np.array([None for i in range(k)])
    total = np.array([0 for i in range(k)])
    for i, image in enumerate(clustering_data):
        accumulated[assignments[i]] = image if accumulated[assignments[i]] is None else accumulated[assignments[i]] + image
        total[assignments[i]] += 1
    centroids = np.array([accumulated[i] / total[i] for i in range(k)])

def reassign_images():
    solved = True
    for i, image in enumerate(clustering_data):
        curr = euclidian_distance(image, centroids[assignments[i]])
        for j, centroid in enumerate(centroids):
            dist = euclidian_distance(centroid, image)
            if curr > dist:
                curr = dist
                assignments[i] = j
                solved = False
    return solved


def cluster():
    global epoch
    solved = False
    while not solved:
        solved = reassign_images()
        reposition_centroids()
        print(epoch); epoch += 1


def test():
    pass


epoch = 0
k = 10  # number of centroids
centroids = np.random.permutation(clustering_data)[0:k]  # fixing k random centroids
assignments = np.array([0 for i in range(len(clustering_data))])  # initial random assignment
cluster()
test()
