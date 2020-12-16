import datetime
import networkx as nx
import time
import argparse

from evaluation.tsne import create_tsne_embedding
from data_input.input import Graph
from node2vec.index import Node2Vec
from evaluation import NodeClassification
from Utils import Params
from LINE.line import LINE
from evaluation.metrics import *


def write_log(log, method, percentage, metrics, time, run_num, dataset):
    log.append(str(run_num) + "," + str(dataset.split("/")[2].split(".")[0]) + "," + str(
        method) + ',' + str(percentage) + ',' + str(metrics['accuracy_un_norm']) + ',' + str(
        metrics['accuracy_norm']) + ',' + str(metrics[
                                                  'precision']) + ',' + str(metrics['recall']) + ',' + str(
        metrics['f1_score_micro']) + ',' + str(metrics['f1_score_macro']) +
               ',' + str(metrics['test_set_size']) + ',' + str(time) + '\n')
    print(log[-1])


def train_and_predict_on_embeddings(embed_, train_, test_, graph):
    n = NodeClassification.NodeClassification()
    n.train(embed_.get_embeddings(train_), graph.get_labels(train_))
    return n.predict(embed_.get_embeddings(test_))


def build_embeddings(method, g, G):
    if method.lower() == 'line':
        l_model = LINE(G, embedding_size=128, proximity_order='all')
        l_model.train(batch_size=1024, epochs=10, verbose=2)
        return l_model.get_embeddings()
    elif method.lower() == 'node2vec':
        node_to_vec = Node2Vec(g)
        node_to_vec.preprocess_transition_probs()
        node_to_vec.simulate_walks()
        return node_to_vec.generate_embeddings()
    else:
        raise Exception("Illegal Args")


percentages = Params.PERCENTS
runs_per_percent = Params.RUNS_PER_PERCENT
parser = argparse.ArgumentParser()
parser.add_argument('-m', '-method', type=str, required=True, help='Method name: Line, DeepWalk, Node2Vec')
parser.add_argument('-d', '-dataset', type=str, required=True, help='Dataset: BlogCatalog, Flickr, Youtube')
parser.add_argument('-r', '-runs', type=int, help='Integer Count of Runs, Default = 10')
print(parser.parse_args())
args = parser.parse_args()

method = args.m
dataset = args.d
if args.r is not None:
    runs_per_percent = args.r

dataset_dict = {'blogcatalog': '../data/blogcatalog.mat', 'flickr': '../data/flickr.mat',
                'youtube': '../data/youtube.mat', 'homosapiens': '../data/homo_sapiens.mat', 'wiki': '../data/wiki.mat'}

datasets = [dataset_dict[dataset.lower()]]

output = []
output.append(
    "Run_num" + "," + "Dataset" + "," + "Method" + ',' + "Percentage" + ',' + "accuracy_un_norm" + ',' + "accuracy_norm" + ',' + "precision" + ',' + "recall" + ',' + "f1_score_micro" + ',' +
    "f1_score_macro" + ',' + "test_set_size" + ',' + "time\n")

for dataset in datasets:
    # Iterate over percentages
    g = Graph(dataset)
    G = nx.Graph(g.get_edges())
    # iterate over run
    for run in range(runs_per_percent):

        s_time = time.time()
        embeddings = build_embeddings(method, g, G)
        e_time = time.time() - s_time

        create_tsne_embedding(embeddings.get_embeddings(g.get_nodes()), g.get_labels(g.get_nodes()), g, dataset,
                              str('_' + method + '_' + str(run)))

        for percent in percentages:
            train, test = g.get_train_test_split(percent)
            predictions = train_and_predict_on_embeddings(embeddings, train, test, g)
            metrics = build_metrics(predictions, g.get_labels(test))
            write_log(output, method, percent * 100, metrics, e_time, run, dataset)

file1 = open(datetime.datetime.now().strftime('output/%Y-%m-%d__%H_%M_%S_%f') + ".csv", 'w')
file1.writelines(output)
file1.close()
