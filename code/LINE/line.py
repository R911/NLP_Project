import random
from utils import *
from .model import create_model
from data_input.input import VertexEmbeddings

class LINE:
    def __init__(self, graph, embedding_size=8, negative_ratio=5, proximity_order='second', ):

        if proximity_order not in ['first', 'second', 'all']:
            raise ValueError('Proximity Order can only be first, second or all')

        self.graph = graph
        self.vertex_embeddings = VertexEmbeddings()
        self.idx_node, self.node_idx = build_node_data(graph)
        self.embedding_size = embedding_size
        self.proximity_order = proximity_order
        self.embeddings = {}
        self.negative_ratio = negative_ratio
        self.node_size = graph.number_of_nodes()
        self.edge_size = graph.number_of_edges()

        self.samples_per_epoch = self.edge_size * (1 + negative_ratio)
        self.use_alias = True
        self.node_prob, self.node_alias = generate_alias_table(
            calculate_vertex_norm_prob(self.graph, self.node_size, self.node_idx))
        self.edge_prob, self.edge_alias = generate_alias_table(calculate_edge_norm_prob(self.graph))

        self.model, self.embedding_dict = create_model(
            self.node_size, self.embedding_size, self.proximity_order)
        self.model.compile('adam', line_loss)
        self.batch_it = self.batch_iterator(self.node_idx)

    def reset_config(self, batch_size, times):
        self.batch_size = batch_size
        self.steps_per_epoch = ((self.samples_per_epoch - 1) // self.batch_size + 1) * times

    def reset_model(self, opt='adam'):

        self.model, self.embedding_dict = create_model(
            self.node_size, self.embedding_size, self.proximity_order)
        self.model.compile(opt, line_loss)
        self.batch_it = self.batch_iterator(self.node_idx)

    def get_embeddings(self, ):
        self.embeddings = {}
        if self.proximity_order == 'first':
            embeddings = self.embedding_dict['first'].get_weights()[0]
        elif self.proximity_order == 'second':
            embeddings = self.embedding_dict['second'].get_weights()[0]
        else:
            embeddings = np.hstack((self.embedding_dict['first'].get_weights()[
                                        0], self.embedding_dict['second'].get_weights()[0]))
        idx_node = self.idx_node
        for i, embedding in enumerate(embeddings):
            self.embeddings[idx_node[i]] = embedding
            self.vertex_embeddings.set_embedding(idx_node[i], embedding)

        return self.vertex_embeddings

    def train(self, batch_size=1024, epochs=5, initial_epoch=0, verbose=1, times=1):
        self.reset_config(batch_size, times)
        history = self.model.fit(self.batch_it, epochs=epochs, initial_epoch=initial_epoch,
                                 steps_per_epoch=self.steps_per_epoch,
                                 verbose=verbose)

        return history

    def batch_iterator(self, node_idx):

        edges = [(node_idx[frm], node_idx[to]) for frm, to in self.graph.edges()]
        data_size = self.edge_size
        shuffle_indices = np.random.permutation(np.arange(data_size))
        mod = 0
        mod_size = 1 + self.negative_ratio
        h = []
        t = []
        sign = 0
        count = 0
        start_index = 0
        end_index = min(start_index + self.batch_size, data_size)
        while True:
            if mod == 0:
                h = []
                t = []
                for i in range(start_index, end_index):
                    if random.random() >= self.edge_prob[shuffle_indices[i]]:
                        shuffle_indices[i] = self.edge_alias[shuffle_indices[i]]
                    cur_h = edges[shuffle_indices[i]][0]
                    cur_t = edges[shuffle_indices[i]][1]
                    h.append(cur_h)
                    t.append(cur_t)
                sign = np.ones(len(h))
            else:
                sign = np.ones(len(h)) * -1
                t = []
                for i in range(len(h)):
                    t.append(get_random_alias_sample(
                        self.node_prob, self.node_alias))

            if self.proximity_order == 'all':
                yield [np.array(h), np.array(t)], [sign, sign]
            else:
                yield [np.array(h), np.array(t)], [sign]
            mod += 1
            mod %= mod_size
            if mod == 0:
                start_index = end_index
                end_index = min(start_index + self.batch_size, data_size)

            if start_index >= data_size:
                count += 1
                mod = 0
                h = []
                shuffle_indices = np.random.permutation(np.arange(data_size))
                start_index = 0
                end_index = min(start_index + self.batch_size, data_size)