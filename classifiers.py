import numpy as np


def softmax(logits, y):
    """
    Computes the loss and gradient for softmax classification.

    Args:
    - logits: A numpy array of shape (N, C)
    - y: A numpy array of shape (N,). y represents the labels corresponding to
    logits, where y[i] is the label of logits[i], and the value of y have a 
    range of 0 <= y[i] < C

    Returns (as a tuple):
    - loss: Loss scalar
    - dlogits: Loss gradient with respect to logits
    """
    loss, dlogits = None, None
    """
    TODO: Compute the softmax loss and its gradient using no explicit loops
    Store the loss in loss and the gradient in dlogits. If you are not careful
    here, it is easy to run into numeric instability. Don't forget the
    normalisation!
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################
    K = np.exp(-np.max(y))
    N, C = logits.shape
    y_one_hot = np.zeros((N, C))
    y_one_hot[np.arange(N), y] = 1.
    soft_logits = np.exp(logits + np.log(K)) / np.exp(logits + np.log(K)).sum(axis=1)[:,None]

    loss = -1/N * np.sum(y_one_hot * np.log(soft_logits))

    dlogits = 1/N * (soft_logits - y_one_hot)
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return loss, dlogits
