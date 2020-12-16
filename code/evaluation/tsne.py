from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import networkx as nx
import time
from Utils import Params

def create_tsne_embedding(embedding, lables, g, dataset="", algo=""):
    dataset_name = dataset.split("/")[2].split(".")[0]
    t = time.time()
    G = nx.Graph(g.get_edges())
    l = []
    e = []
    for key in G.nodes:
        l.append(lables[key])
        e.append(embedding[key])
    model = TSNE(n_components=2, n_jobs=Params.CORES)
    embeding_position = model.fit_transform(e)
    # plt.scatter(embeding_position[:, 0], embeding_position[:, 1], c=l)
    plt.figure(clear=True)
    nx.draw_networkx_edges(G, embeding_position,
                           width=0.01,
                           arrows=False, alpha=0.5)
    nx.draw_networkx_nodes(G, embeding_position, node_color=l,
                           node_size=1)
    plt.title("embedding for {} {}".format(dataset_name, algo))
    plt.savefig("./output/embedding_{}_{}.png".format(dataset_name, algo),dpi=1000)
    print("Created TSNE in {} seconds ".format(time.time() - t))
