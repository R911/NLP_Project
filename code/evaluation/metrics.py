from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score


def build_metrics(pred_dict, true_dict):
    y_true = []
    y_pred = []
    for key in true_dict:
        y_pred.append(pred_dict.get(key))
        y_true.append(true_dict.get(key))

    return calculate_metrics(y_true, y_pred)


def calculate_metrics(y_true, y_pred):
    results = {'accuracy_un_norm': accuracy_score(y_true, y_pred, normalize=False),
               'accuracy_norm': accuracy_score(y_true, y_pred, normalize=True),
               'precision': precision_score(y_true, y_pred, average='weighted', zero_division=1),
               'f1_score_micro': f1_score(y_true, y_pred, average='micro'),
               'f1_score_macro': f1_score(y_true, y_pred, average='macro'),
               'recall': recall_score(y_true, y_pred,average='weighted'),
               'test_set_size': len(y_true)}

    return results
