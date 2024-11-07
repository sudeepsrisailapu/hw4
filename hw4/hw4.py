import csv
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram


def load_data(filepath):
    file_data = []
    with open(filepath, mode='r') as file:
        file_reader = csv.DictReader(file)

        for row in file_reader:
            file_data.append(dict(row))

    return file_data


def calc_features(row):
    try:
        x_one = float(row['Population'])
        x_two = float(row['Net migration'])
        x_three = float(row['GDP ($ per capita)'])
        x_four = float(row['Literacy (%)'])
        x_five = float(row['Phones (per 1000)'])
        x_six = float(row['Infant mortality (per 1000 births)'])
    except KeyError as e:
        raise ValueError("Could not find field: {e}")
    except ValueError as e:
        raise ValueError("Could not convert value: {e}")

    features = np.array([x_one, x_two, x_three, x_four, x_five, x_six], dtype=np.float64)

    return features


def hac(features):
    n = len(features)

    distance_matrix = np.full((2 * n, 2 * n), math.inf)

    for i in range(n):
        for j in range(i + 1, n):
            distance_matrix[i, j] = distance_matrix[j, i] = np.linalg.norm(features[i] - features[j])

    cluster = {i: [i] for i in range(n)}
    output_array = np.zeros((n - 1, 4))
    next_cluster = n

    for k in range(n - 1):
        index = np.argmin(distance_matrix[:n + k, :n + k])
        r, c = np.unravel_index(index, distance_matrix[:n + k, :n + k].shape)

        if r > c:
            r, c = c, r
        for i in range(n + k):
            if i != r and i != c:
                distance_matrix[i, n + k] = distance_matrix[n + k, i] = min(distance_matrix[i, r], distance_matrix[i, c])

        cluster[next_cluster] = cluster[r] + cluster[c]

        output_array[k, 0] = r
        output_array[k, 1] = c
        output_array[k, 2] = distance_matrix[r, c]
        output_array[k, 3] = len(cluster[next_cluster])

        distance_matrix[r, :] = math.inf
        distance_matrix[:, r] = math.inf
        distance_matrix[c, :] = math.inf
        distance_matrix[:, c] = math.inf

        next_cluster += 1

    return output_array


def fig_hac(Z, names):
    figure = plt.figure(figsize=(10, 7))

    dendrogram(Z, labels=names, leaf_rotation=90)

    plt.tight_layout()

    plt.show()

    return figure


def normalize_features(features):
    feature_matrix = np.array(features)

    min_columns = np.min(feature_matrix, axis=0)
    max_columns = np.max(feature_matrix, axis=0)

    normal_matrix = (feature_matrix - min_columns) / (max_columns - min_columns)

    normal_features = [np.array(row, dtype=np.float64) for row in normal_matrix]

    return normal_features
