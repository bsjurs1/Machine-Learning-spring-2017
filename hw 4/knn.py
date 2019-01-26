"""Assigment 4."""

import sys
if sys.version_info >= (3, 0):
    from queue import PriorityQueue
else:
    from Queue import PriorityQueue
import numpy as np
from math import *

inf = float('inf')

k = int(sys.argv[1])
train_data_file_path = sys.argv[2]
test_data_file_path = sys.argv[3]


def get_data(data_file_path):
    """Extract input from filepath to standard format."""
    data_file = open(data_file_path, 'r').readlines()
    data = []
    n = -1
    dim = -1
    for i in range(len(data_file)):
        line_elems = [float(x) for x in data_file[i].split()]
        if i == 0:
            n = int(line_elems[0])
            dim = int(line_elems[1])
        else:
            data.append(np.array(line_elems))
    return data, n, dim


train_data, train_n, train_dim = get_data(train_data_file_path)
test_data, test_n, test_dim = get_data(test_data_file_path)


def get_k_nearest_neigbours(test_point, train_data):
    """Retrieve the k nearest neighbors of test point."""
    # Priority queue sorted on 1/distance, such that the largest distance
    # neighbours are on the top of the queue

    k_nearest_neighbours = PriorityQueue(k)

    # Determine k nearest neighbours
    for train_point in train_data:
        dist = np.linalg.norm(test_point - train_point[0:train_dim])
        inverse_dist = float(1.0) / float(dist)
        if not k_nearest_neighbours.full():
            k_nearest_neighbours.put((inverse_dist, train_point))
        if k_nearest_neighbours.queue[0][0] < inverse_dist:
            k_nearest_neighbours.get()
            k_nearest_neighbours.put((inverse_dist, train_point))

    return k_nearest_neighbours


def get_votes(k_nearest_neighbours):
    """Get the votes of the k nearest neighbours."""
    votes = {}
    # Do voting over classes
    for neighbour in k_nearest_neighbours.queue:
        neighbour = neighbour[1]
        neighbour_class = neighbour[train_dim]
        if neighbour_class not in votes.keys():
            votes[neighbour_class] = 0
        votes[neighbour_class] += 1
    return votes


def get_max_vote(votes):
    """Get the maximum voted labels."""
    max_vote = 0
    candidate_classes = []
    for neighbour_class in votes.keys():
        vote_count = votes[neighbour_class]
        if vote_count > max_vote:
            max_vote = vote_count
            candidate_classes = []
            candidate_classes.append(neighbour_class)
        elif vote_count == max_vote:
            candidate_classes.append(neighbour_class)
    return candidate_classes


def get_closest_points(test_point, k_nearest_neighbours, max_votes):
    """Get the closest points to test point."""
    min_dist = inf
    candidate_points = []
    for neighbour in k_nearest_neighbours.queue:
        neighbour = neighbour[1]
        neighbour_class = neighbour[train_dim]
        if neighbour_class in max_votes:
            dist = np.linalg.norm(test_point -
                                  neighbour[0:train_dim])
            if dist < min_dist:
                min_dist = dist
                candidate_points = []
                candidate_points.append(neighbour)
            elif dist == min_dist:
                candidate_points.append(neighbour)
    return candidate_points


def get_lowest_label(candidate_points):
    """Find the lowest label among candidate points."""
    lowest_label = inf
    for candidate_instance in candidate_points:
        label = candidate_instance[train_dim]
        if label < lowest_label:
            lowest_label = label
    return lowest_label


def get_test_point_label(test_point, max_votes, k_nearest_neighbours):
    """Return the label of the test point."""
    test_point_label = -1
    if len(max_votes) == 1:
        test_point_label = max_votes[0]
    else:
        # Find closest points
        candidate_points = get_closest_points(test_point, k_nearest_neighbours,
                                              max_votes)

        if len(candidate_points) == 1:
            test_point_label = candidate_points[0][train_dim]
        else:
            test_point_label = get_lowest_label(candidate_points)

    return test_point_label


def knn(test_point, train_data):
    """Use k nearest neighbours to predict label of test point."""
    k_nearest_neighbours = get_k_nearest_neigbours(test_point, train_data)

    votes = get_votes(k_nearest_neighbours)

    max_votes = get_max_vote(votes)

    test_point_label = get_test_point_label(test_point, max_votes,
                                            k_nearest_neighbours)

    return test_point_label


for i in range(test_n):
    test_point = test_data[i]
    test_point_label = knn(test_point, train_data)
    print(str(i + 1) + ". " + ' '.join(map(str, test_point)) + " -- " +
          str(test_point_label))
