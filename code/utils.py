import numpy as np
import math
from tensorflow.python.keras import backend as kb


def line_loss(y_true, y_pred):
    return -kb.mean(kb.log(kb.sigmoid(y_true * y_pred)))


def build_node_data(graph):
    node_idx = {}
    idx_node = []
    n = 0
    for node in graph.nodes():
        node_idx[node] = n
        idx_node.append(node)
        n += 1
    return idx_node, node_idx


def calculate_vertex_norm_prob(graph, node_size, node_idx):
    node_degree = np.zeros(node_size)
    for edge in graph.edges():
        node_degree[node_idx[edge[0]]] += graph[edge[0]][edge[1]].get('weight', 1.0)

    total_sum = sum([math.pow(node_degree[i], 0.75)
                     for i in range(node_size)])
    norm_prob = [float(math.pow(node_degree[j], 0.75)) /
                 total_sum for j in range(node_size)]

    return norm_prob


def calculate_edge_norm_prob(graph):
    numEdges = graph.number_of_edges()
    total_sum = sum([graph[edge[0]][edge[1]].get('weight', 1.0)
                     for edge in graph.edges()])
    norm_prob = [graph[edge[0]][edge[1]].get('weight', 1.0) *
                 numEdges / total_sum for edge in graph.edges()]

    return norm_prob


"""
Alias Sampling
Vose's Alias Method
https://www.keithschwarz.com/darts-dice-coins/
"""


def generate_alias_table(probs):
    n = len(probs)
    prob = [0] * n
    alias = [0] * n
    small = []
    large = []
    scaled_prob = np.array(probs) * n
    for i, probs in enumerate(scaled_prob):
        if probs < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        small_idx = small.pop()
        large_idx = large.pop()
        prob[small_idx] = scaled_prob[small_idx]
        alias[small_idx] = large_idx
        scaled_prob[large_idx] = scaled_prob[large_idx] + scaled_prob[small_idx] - 1
        if scaled_prob[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)

    while large:
        large_idx = large.pop()
        prob[large_idx] = 1
    while small:
        small_idx = small.pop()
        prob[small_idx] = 1

    return prob, alias


def get_random_alias_sample(prob, alias):
    N = len(prob)
    i = int(np.random.random() * N)
    r = np.random.random()
    if r < prob[i]:
        return i
    else:
        return alias[i]
