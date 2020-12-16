from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from Utils import Params

class NodeClassification:
    def __init__(self):
        self.m = LogisticRegression(max_iter=750) # default is 100
        self.model = OneVsRestClassifier(self.m, n_jobs=Params.CORES)

    def train(self, embeddings, classes):
        embeding_list= []
        classes_list = []
        for node in embeddings:
            embeding_list.append(embeddings[node])
            classes_list.append(classes[node])
        # print(np.array(embeding_list))
        # print(np.array(classes_list))
        self.model.fit(np.array(embeding_list), np.array(classes_list))

    def predict(self, embeddings):
        embedding_list= []
        embeddings_keys = list(embeddings.keys())
        for node in range(len(embeddings_keys)):
            embedding_list.append(embeddings[embeddings_keys[node]])
        predictions = self.model.predict(np.array(embedding_list))
        predictions_dict = {}
        for node in range(len(embeddings_keys)):
            predictions_dict[embeddings_keys[node]] = predictions[node]
        return predictions_dict
