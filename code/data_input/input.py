from scipy.io import loadmat
import random
import math


class Graph:

  def __init__(self, data): #TODO: Percent
    self.graph = {}
    self.labels = {}
    self.title = data.split(".mat")[0].split('/')[-1]
    self.read_mat_matrix(data)

  def set_random_seed(self, seed):
    random.seed(seed)

  def neighbors(self, node):
    return self.graph[node]

  def from_edges(self, edges):
    for edge in edges:
      from_v, to_v = edge

      if from_v not in self.graph:
        self.graph[from_v] = []

      self.graph[from_v].append(to_v)

  def get_nodes(self):
    return list(self.graph.keys())

  def order(self):
    return len(self.get_nodes())

  def has_edge(self, frm, to):
    return to in self.graph[frm]

  def get_edges(self):
    return [(from_node, to_node) for from_node, to_node_list in self.graph.items() for to_node in to_node_list]

  def read_mat_matrix(self, filepath):

    matfile = loadmat(filepath)

    network = matfile['network'].tocoo()
    self.from_edges(zip(network.row, network.col))

    groups = matfile['group'].tocoo()
    self.labels = {node: label for node, label in zip(groups.row, groups.col)}

  def rand_subset_k_nodes(self, k):
    random.seed()
    return random.sample(self.graph.keys(), k)

  def get_train_test_split(self, train_pct):

    # get random order of nodes
    nodes = self.rand_subset_k_nodes(self.order())

    #get index into training nodes
    num_train_nodes = int(math.floor(train_pct * self.order()))

    return nodes[:num_train_nodes], nodes[num_train_nodes:]

  def get_random_walk(self, inital_vertex, length):
    # TODO: make sure seed is set
    random.seed()
    list_of_vertexes = [str(inital_vertex)]
    current_vertex = inital_vertex
    for i in range(length):
      current_vertex = random.sample(self.graph[current_vertex], 1)[0]
      list_of_vertexes.append(str(current_vertex))
    return list_of_vertexes

  def get_nodes_in_random_order(self):
    return random.sample(self.get_nodes(), len(self.get_nodes()))

  def get_labels(self, nodes):
    return {node: self.labels[node] for node in nodes}


class VertexEmbeddings:

  def __init__(self):
    self.vertex_embeddings = {}

  # Set embedding for a particular node.
  def set_embedding(self, node, embedding):
    self.vertex_embeddings[node] = embedding

  
  # List of node names to retrieve embeddings for
  def get_embeddings(self, nodes):
    return {node: self.vertex_embeddings[node] for node in nodes}


# g = Graph()
# g.read_mat_matrix('blogcatalog.mat')
# print(g.get_train_test_split(0.001)["train"])
