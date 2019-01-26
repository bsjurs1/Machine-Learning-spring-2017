"""Assigment 5."""

import sys
import numpy as np
import math

T = int(sys.argv[1])
train_pos_file_path = sys.argv[2]
train_neg_file_path = sys.argv[3]
test_pos_file_path = sys.argv[4]
test_neg_file_path = sys.argv[5]


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


def binary_classifier(train_data, dim, wi):
    """Compute the binary classifier based on the weighted mean."""
    n = np.zeros(dim)
    p = np.zeros(dim)
    p_w = 0
    n_w = 0
    for i in range(len(train_data)):
        if train_data[i][dim] == 1:
            # Positive
            p_w += float(wi[i])
            p += (float(wi[i]) * train_data[i][0:dim])
        elif train_data[i][dim] == -1:
            # Negative
            n_w += float(wi[i])
            n += (float(wi[i]) * train_data[i][0:dim])

    p *= float(1) / float(p_w)
    n *= float(1) / float(n_w)
    w_vec = p - n
    t_vec = 0.5 * np.dot(np.transpose(p + n), (p - n))

    error = 0

    for i in range(len(train_data)):
        point = train_data[i]
        # Predicted positive
        if np.dot(point[0:dim], w_vec) > t_vec:
            if point[dim] == -1:
                # It is a false positive
                error += wi[i]
        # Predicted negative
        else:
            if point[dim] == 1:
                # It is a false positive
                error += wi[i]

    return t_vec, w_vec, error


def boosting(train_data, dim, t):
    """Train an ensemble of reweighted classifiers from training sets."""
    w = []
    w.append([float(1) / float(len(train_data))] * len(train_data))

    # Store models in m, models are stored as a tuple with the w_vector as well
    # as the t_vector

    m = []

    for i in range(t):
        print("Iteration " + str(i + 1) + str(":"))
        t_vec, w_vec, error = binary_classifier(train_data, dim, w[i])
        alpha = 0.5 * math.log(float(1 - error) / float(error))
        print("Error = " + str(error))
        print("Alpha = " + str(alpha))
        if error >= 0.5:
            break
        # Add model only if it has error rate less than 0.5
        m.append((t_vec, w_vec, alpha))

        is_increase_weights_printed = False
        is_decrease_weights_printed = False
        factor_to_increase = 0
        factor_to_decrease = 0
        # Update weights by figuring out which points that are misclassified
        w.append([0] * len(train_data))
        for j in range(len(train_data)):
            if np.dot(train_data[j][0:dim], w_vec) > t_vec:
                if train_data[j][dim] == -1:
                    # misclassified
                    w[i + 1][j] = float(w[i][j]) / float(2 * error)
                    if not is_increase_weights_printed:
                        factor_to_increase = float(1) / float(2 * error)
                        is_increase_weights_printed = True
                else:
                    # correctly classified
                    w[i + 1][j] = float(w[i][j]) / float(2 * (1 - error))
                    if not is_decrease_weights_printed:
                        factor_to_decrease = float(1) / float(2 * (1 - error))
                        is_decrease_weights_printed = True
            else:
                if train_data[j][dim] == 1:
                    # misclassified
                    w[i + 1][j] = float(w[i][j]) / float(2 * error)
                    if not is_increase_weights_printed:
                        factor_to_increase = float(1) / float(2 * error)
                        is_increase_weights_printed = True
                else:
                    # correctly classified
                    w[i + 1][j] = float(w[i][j]) / float(2 * (1 - error))
                    if not is_decrease_weights_printed:
                        factor_to_decrease = float(1) / float(2 * (1 - error))
                        is_decrease_weights_printed = True

        print("Factor to increase weights = " + str(factor_to_increase))
        print("Factor to decrease weights = " + str(factor_to_decrease))

    return m


train_pos_data, train_pos_n, train_pos_dim = get_data(train_pos_file_path)
train_neg_data, train_neg_n, train_neg_dim = get_data(train_neg_file_path)
test_pos_data, test_pos_n, test_pos_dim = get_data(test_pos_file_path)
test_neg_data, test_neg_n, test_neg_dim = get_data(test_neg_file_path)

for i in range(train_pos_n):
    train_pos_data[i] = np.append(train_pos_data[i], 1)

for i in range(train_neg_n):
    train_neg_data[i] = np.append(train_neg_data[i], -1)

train_pos_data.extend(train_neg_data)

m = boosting(train_pos_data, train_pos_dim, T)

FP = 0
FN = 0

for point in test_pos_data:
    alpha_summation = 0
    for t_vec, w_vec, alpha in m:
        classification = np.dot(w_vec, point) - t_vec
        alpha_summation += alpha * np.sign(classification)
    if not alpha_summation > 0:
        # negative
        FN += 1

for point in test_neg_data:
    alpha_summation = 0
    for t_vec, w_vec, alpha in m:
        classification = np.dot(w_vec, point) - t_vec
        alpha_summation += alpha * np.sign(classification)
    if alpha_summation > 0:
        # positive
        FP += 1




print("False positives: " + str(FP))
print("False negatives: " + str(FN))
print("Error rate: " + str(float(FP + FN) / (len(test_pos_data) +
                                             len(test_neg_data))))
