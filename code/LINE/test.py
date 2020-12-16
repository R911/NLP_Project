from ..data_input.input import Graph
import networkx as nx
from ..LINE.line import LINE

if __name__ == "__main__":
    g = Graph('../../data/blogcatalog.mat')
    G = nx.Graph(g.get_edges())
    model = LINE(G, embedding_size=64, proximity_order='all')
    model.train(batch_size=1024, epochs=2, verbose=2)
    embeddings = model.get_embeddings()