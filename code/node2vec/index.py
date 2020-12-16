from data_input.input import VertexEmbeddings
from utils import generate_alias_table, get_random_alias_sample
from Utils import Params

import random
import time
from gensim.models import Word2Vec
import json


class Node2Vec:

    def __init__(self,
                 g,
                 dimensions=128,
                 walk_length=30,
                 num_walks=30,
                 window_size=10,
                 inout_param=0.25,
                 return_param=0.25):

        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.inout_param = inout_param
        self.return_param = return_param
        self.alias_nodes = {}
        self.alias_edges = {}
        self.walks = []
        self.g = g
        self.title = "{}_nw{}_wl{}_io{}_re{}".format(g.title, num_walks, walk_length, int(inout_param * 100), int(return_param * 100))
        self.vertex_embeddings = VertexEmbeddings()

    def generate_embeddings(self):

        walks = [list(map(str, walk)) for walk in self.walks]
        start = time.time()
        print("STARTING TO TRAIN WORD 2 VEC")
        model = Word2Vec(walks, size=self.dimensions, window=self.window_size, min_count=0, sg=1, workers=Params.CORES)
        print("FINISHED TRAINING WORD 2 VEC: {} SECONDS".format(time.time() - start))

        for node in self.g.get_nodes():
            self.vertex_embeddings.set_embedding(node, model.wv.get_vector(str(node)))

        # model.wv.save_word2vec_format('emb/{}.emb'.format(self.title))
        return self.vertex_embeddings

    def preprocess_transition_probs(self):

        start = time.time()
        print("STARTING TO PREPROCESS TRANSITION PROBABILITIES")
        # This code basically preprocesses the network and calculates transition probabilities between nodes.
        for node in self.g.get_nodes():
            len_nbr = len(self.g.neighbors(node))
            self.alias_nodes[node] = generate_alias_table([1 / len_nbr] * len_nbr)

        for edge in self.g.get_edges():
            frm, to = edge
            # Instead of assuming a uniform probability distribution among neighbors, bias the transition probabilities
            # according to the in-out and return hyperparameters.
            self.alias_edges[edge] = self.get_alias_edge(frm, to)
            self.alias_edges[(to, frm)] = self.get_alias_edge(to, frm)

        print("FINISHED PREPROCESSING TRANSITION PROBABILITIES IN : {} SECONDS".format(time.time() - start))

    def get_alias_edge(self, src, dst):

        probs = []
        for nbr in sorted(self.g.neighbors(dst)):
            if nbr == src:
                probs.append(1 / self.return_param)
            elif self.g.has_edge(nbr, src):
                probs.append(1)
            else:
                probs.append(1 / self.inout_param)

        total = sum(probs)
        return generate_alias_table([float(prob) / total for prob in probs])

    def simulate_walks(self):
        nodes = list(self.g.get_nodes())
        for walk_iter in range(self.num_walks):
            print("WALK ITERATION {}".format(walk_iter))
            random.shuffle(nodes)
            for node in nodes:
                self.walks.append(self.node2vec_walk(node))

        # with open('walks/{}.txt'.format(self.title), 'w') as outfile:
        #     json.dump(list(map(str, self.walks)), outfile)

    def node2vec_walk(self, start_node):
        walk = [start_node]

        while len(walk) < self.walk_length:
            cur = walk[-1]
            nbrs = sorted(self.g.neighbors(cur))
            if len(nbrs) > 0:
                if len(walk) == 1:
                    probs, alias = self.alias_nodes[cur]
                else:
                    prev = walk[-2]
                    probs, alias = self.alias_edges[(prev, cur)]
                walk.append(nbrs[get_random_alias_sample(probs, alias)])
            else:
                break

        return walk
