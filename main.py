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
        if solved:
            break
        reposition_centroids()
        print("iteration", epoch); epoch += 1
    print("clustering completed")


def test():
    print("testing...", end="")
    incorrect = 0
    for image, label in zip(testing_data.images, testing_data.labels):
        min_dist = np.inf
        predicted = 0
        for i, centroid in enumerate(centroids):
            dist = euclidian_distance(image, centroid)
            if dist < min_dist:
                min_dist = dist
                predicted = i
        predicted = assigned_class[predicted]
        if predicted != label:
            incorrect += 1
    print('done' + '\n' + "accuracy:", 100 * (len(data.test.images) - incorrect) / len(data.test.images))


def assign_classes():
    global assigned_class
    result = [[0 for j in range(k)] for i in range(10)]
    for index, image in enumerate(clustering_data):
        d = np.inf
        pred = 0
        for i, centroid in enumerate(centroids):
            curr_dist = euclidian_distance(image, centroid)
            if curr_dist < d:
                d = curr_dist
                pred = i
        result[data.train.labels[index]][pred] += 1

    # assign each centroid a class
    for i in range(k):
        # solve for centroid i
        max_index = 0
        max_val = -np.inf
        for j in range(10):
            if result[j][i] > max_val:
                max_val = result[j][i]
                max_index = j
        assigned_class[i] = max_index

    print(assigned_class)


if __name__ == '__main__':
    epoch = 0
    k = 50  # number of centroids
    centroids = np.random.permutation(clustering_data)[0:k]  # fixing k random centroids
    assignments = np.array([0 for i in range(len(clustering_data))])  # initial random assignment
    assigned_class = [0 for i in range(k)]
    cluster()
    assign_classes()
    test()
