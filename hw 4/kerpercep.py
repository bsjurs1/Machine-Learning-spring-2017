"""Assigment 2."""

import sys
import numpy as np
from math import *


sigma = float(sys.argv[1])
p_train_data_file_path = sys.argv[2]
n_train_data_file_path = sys.argv[3]
p_test_data_file_path = sys.argv[4]
n_test_data_file_path = sys.argv[5]


def get_data(data_file_path):
    """Extract input from filepath to standard format."""
    data_file = open(data_file_path, 'r').readlines()
    data = []
    k = -1
    dim = -1
    for i in range(len(data_file)):
        line_elems = [float(x) for x in data_file[i].split()]
        if i == 0:
            k = int(line_elems[0])
            dim = int(line_elems[1])
        else:
            line_elems.append(1)
            data.append(np.array(line_elems))
    return data, k, dim


def kernel(xi, xj):
    """Calculate the gaussian kernel."""
    return exp(-float(np.linalg.norm((xi - xj))**2) / float(2 * (sigma**2)))


def kernel_perceptron(d, ys):
    """Train a kernel perceptron."""
    alphas = [0.0] * len(d)
    converged = False

    while not converged:
        converged = True
        for i in range(len(d)):

            cum_sum = 0
            for j in range(len(d)):
                cum_sum += (alphas[j] * ys[j] * kernel(d[i], d[j]))

            if ys[i] * cum_sum <= 0:
                alphas[i] += 1
                converged = False

    return alphas


p_train_data, p_train_k, p_train_dim = get_data(p_train_data_file_path)
n_train_data, n_train_k, n_train_dim = get_data(n_train_data_file_path)
p_test_data, p_test_k, p_test_dim = get_data(p_test_data_file_path)
n_test_data, n_test_k, n_test_dim = get_data(n_test_data_file_path)

ys = [1] * len(p_train_data)
ys.extend([-1] * len(n_train_data))
p_train_data.extend(n_train_data)

alphas = kernel_perceptron(p_train_data, ys)

P = len(p_test_data)
N = len(n_test_data)

FP = 0
TP = 0

for p_test_point in p_test_data:

    cum_sum = 0
    for i in range(len(alphas)):
        cum_sum += (alphas[i] * ys[i] * kernel(p_test_point, p_train_data[i]))

    if(cum_sum > 0):
        # positive
        TP += 1
    else:
        # negative
        FP += 1


FN = 0
TN = 0

for n_test_point in n_test_data:

    cum_sum = 0
    for i in range(len(alphas)):
        cum_sum += (alphas[i] * ys[i] * kernel(n_test_point, p_train_data[i]))

    if(cum_sum > 0):
        # positive
        FN += 1
    else:
        # negative
        TN += 1

print("Alphas: " + ' '.join(map(str, alphas)))
print("False positives: " + str(FP))
print("False negatives: " + str(FN))
print("Error rate: " + str(100 * float(FP + FN) / float(P + N)) + str("%"))
