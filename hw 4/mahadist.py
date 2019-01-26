"""Assigment 1."""

import sys
import numpy as np
from math import *

train_data_file_path = sys.argv[1]
test_data_file_path = sys.argv[2]

train_data_file = open(train_data_file_path, 'r')
test_data_file = open(test_data_file_path, 'r')

train_data_file = train_data_file.readlines()
test_data_file = test_data_file.readlines()

train_data = []
k = -1
train_dim = -1
for i in range(len(train_data_file)):
    line_elems = [float(x) for x in train_data_file[i].split()]
    if i == 0:
        k = int(line_elems[0])
        train_dim = int(line_elems[1])
    else:
        train_data.append(np.array(line_elems))

test_data = []
test_k = -1
test_dim = -1
for i in range(len(test_data_file)):
    line_elems = [float(x) for x in test_data_file[i].split()]
    if i == 0:
        test_k = int(line_elems[0])
        test_dim = int(line_elems[1])
    else:
        test_data.append(np.array(line_elems))

centroid = np.divide(np.sum(train_data, axis=0), k)
xzt = np.subtract(train_data, centroid)
xz = np.transpose(xzt)
s = np.dot(xz, xzt)
cov_matrix = np.divide(s, k)
inverse_cov_matrix = np.linalg.inv(cov_matrix)

print("Centroid: " + ' '.join(map(str, centroid)))

print("Covariance matrix: ")
print(str(cov_matrix))

print("Distances:")
for i in range(len(test_data)):
    point = test_data[i]
    mahab_dist = sqrt(np.dot(np.dot(np.transpose(np.subtract(point, centroid)),
                                    inverse_cov_matrix), np.subtract(point,
                                                                     centroid)))
    print(str(i + 1) + ". " + ' '.join(map(str, point)) + " -- " +
          str(mahab_dist))
