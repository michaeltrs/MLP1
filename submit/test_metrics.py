import numpy as np

def count(actual_targets, predicted_targets, i, j):
    """
    count occurences where actual_target is i and predicted_target is j
    """
    idx_i = actual_targets == i
    return (predicted_targets[idx_i] == j).sum()


def confusion_mat(actual_targets, predicted_targets, norm=False):
    """
    Confusion Matrix for N-class classification
    If norm is True a normalized (per actual class) matrix is returned
    """
    assert len(predicted_targets) == len(actual_targets)
    labels = np.unique(np.concatenate((predicted_targets, actual_targets))) # all unique labels
    conf_mat = np.array(
        [[count(actual_targets, predicted_targets, i, j) for j in labels] for i in labels])
    if norm:
        conf_mat = (conf_mat.T / conf_mat.sum(axis=1)).T
    return conf_mat  


def recall_precision(actual_targets, predicted_targets, percent=False):
    """
    For binary classification
    """
    assert len(predicted_targets) == len(actual_targets)
    # indices fro positives and negatives
    neg_idx = predicted_targets == 0
    pos_idx = predicted_targets == 1
    # TP, FP, FN, TN
    true_pos = float((actual_targets[pos_idx] == 1).sum())
    false_pos = float((actual_targets[pos_idx] == 0).sum())
    false_neg = float((actual_targets[neg_idx] == 1).sum())
    # true_neg = (actual_targets[neg_idx] == 0).sum()
    recall = true_pos / (true_pos + false_neg)
    precision = true_pos / (true_pos + false_pos)
    if percent:
        recall *= 100
        precision *= 100
    return recall, precision

def f_score(recall, precision, a=1):
    """
    Fa_score from precision and recall
    """
    return (1 + a) * (precision * recall) / (a * precision + recall)
