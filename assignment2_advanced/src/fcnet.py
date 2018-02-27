import numpy as np

from src.classifiers import softmax
from src.layers import (linear_forward, linear_backward, relu_forward,
                        relu_backward, dropout_forward, dropout_backward)



def random_init(n_in, n_out, weight_scale=5e-2, dtype=np.float32):
    """
    Weights should be initialized from a normal distribution with standard
    deviation equal to weight_scale and biases should be initialized to zero.

    Args:
    - n_in: The number of input nodes into each output.
    - n_out: The number of output nodes for each input.
    """
    W = None
    b = None
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################
    W = weight_scale * np.random.randn(n_in, n_out)
    b = np.zeros(n_out)
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return W, b



class FullyConnectedNet(object):
    """
    Implements a fully-connected neural networks with arbitrary size of
    hidden layers. For a network with N hidden layers, the architecture would
    be as follows:
    [linear - relu - (dropout)] x (N - 1) - linear - softmax
    The learnable params are stored in the self.params dictionary and are
    learned with the Solver.
    """
    def __init__(self, hidden_dims, input_dim=32*32*3, num_classes=10,
                 dropout=0, reg=0.0, weight_scale=1e-2, dtype=np.float32,
                 seed=None):
        """
        Initialise the fully-connected neural networks.
        Args:
        - hidden_dims: A list of the size of each hidden layer
        - input_dim: An integer giving the size of the input
        - num_classes: Number of classes to classify.
        - dropout: A scalar between 0. and 1. determining the dropout factor.
        If dropout = 0., then dropout is not applied.
        - reg: Regularisation factor.

        """
        self.dtype = dtype
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.use_dropout = True if dropout > 0.0 else False
        if seed:
            np.random.seed(seed)
        self.params = dict()
        """
        TODO: Initialise the weights and bias for all layers and store all in
        self.params. Store the weights and bias of the first layer in keys
        W1 and b1, the weights and bias of the second layer in W2 and b2, etc.
        Weights and bias are to be initialised according to the random
        initialisation (see manual).
        """
        #######################################################################
        #                           BEGIN OF YOUR CODE                        #
        #######################################################################
        # input -> first hidden layer
        self.params["W1"], self.params["b1"] = \
            random_init(input_dim, hidden_dims[0], weight_scale, dtype)
        #print("shape of W1: ", self.params["W1"].shape)
        #print("shape of b1: ", self.params["b1"].shape)
        # hidden layer i-1 -> hidden layer i
        for l in range(1, self.num_layers-1):
            self.params["W%d"%(l+1)],  self.params["b%d"%(l+1)] = \
                random_init(hidden_dims[l-1], hidden_dims[l], weight_scale, dtype)
            #print("shape of W%d: "%(l+1), self.params["W%d"%(l+1)].shape)
            #print("shape of b%d: "%(l+1), self.params["b%d"%(l+1)].shape)
        # last hidden layer -> output
        l += 1
        self.params["W%d" % (l+1)], self.params["b%d" % (l+1)] = \
            random_init(hidden_dims[l-1], num_classes, weight_scale, dtype)
        #print("shape of W%d: " % (l + 1), self.params["W%d"%(l+1)].shape)
        #print("shape of b%d: " % (l + 1), self.params["b%d"%(l+1)].shape)
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
        # When using dropout we need to pass a dropout_param dictionary to
        # each dropout layer so that the layer knows the dropout probability
        # and the mode (train / test). You can pass the same dropout_param to
        # each dropout layer.
        self.dropout_params = dict()
        if self.use_dropout:
            self.dropout_params = {"train": True, "p": dropout}
            if seed is not None:
                self.dropout_params["seed"] = seed
        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
        # print(self.dropout_params)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
        Args:
        - X: Input data, numpy array of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
        Returns:
        If y is None, then run a test-time forward pass of the model and
        return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
        If y is not None, then run a training-time forward and backward pass
        and return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping
        parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        X = X.astype(self.dtype)
        linear_cache = dict()
        relu_cache = dict()
        dropout_cache = dict()
        """
        TODO: Implement the forward pass for the fully-connected neural
        network, compute the scores and store them in the scores variable.
        """
        #######################################################################
        #                           BEGIN OF YOUR CODE                        #
        #######################################################################

        # input -> first hidden layer
        if self.use_dropout:
            input_next, mask = dropout_forward(X,  p=self.dropout_params["p"],
                train=self.dropout_params["train"], seed=self.dropout_params["seed"])
            dropout_cache[1] = mask
            #print("shape of dropout to layer 1: ", input_next.shape)
            if y is not None:
                input_next = (1 - self.dropout_params["p"]) * activation_out
        else:
            input_next = X.copy()
            #print("shape of input to layer 1: ", input_next.shape)
        linear_cache[1] = linear_forward(input_next, self.params["W1"], self.params["b1"])
        #print("shape of linear cache 1: ", linear_cache[1].shape)
        activation_out = relu_forward(linear_cache[1])
        relu_cache[1] = activation_out
        #print("shape of relu cache 1: ", relu_cache[1].shape)

        # hidden layer i -> hidden layer i+1
        for l in range(2, self.num_layers):
            if self.use_dropout:
                input_next, mask = dropout_forward(activation_out, p=self.dropout_params["p"],
                    train=self.dropout_params["train"], seed=self.dropout_params["seed"])
                dropout_cache[l] = mask
                #print("shape of input to layer %d: "%l, input_next.shape)
            else:
                input_next = activation_out.copy()
                #print("shape of input to layer %d: " % l, input_next.shape)
            linear_cache[l] = linear_forward(input_next, self.params["W%d"%l], self.params["b%d"%l])
            #print("shape of linear cache %d: " % l, linear_cache[l].shape)
            activation_out = relu_forward(linear_cache[l])
            relu_cache[l] = activation_out
            #print("shape of relu cache %d: " % l, relu_cache[l].shape)

        # last hidden layer -> output layer
        l += 1
        if self.use_dropout:
            input_next, mask = dropout_forward(activation_out, p=self.dropout_params["p"],
                train=self.dropout_params["train"], seed=self.dropout_params["seed"])
            dropout_cache[l] = mask
            #print("shape of dropout to layer %d: " % l, input_next.shape)
        else:
            input_next = activation_out.copy()
            #print("shape of input to layer %d: " % l, input_next.shape)
        linear_cache[l] = linear_forward(input_next, self.params["W%d"%l], self.params["b%d"%l])
        # activation_out = relu_forward(linear_cache[l])
        # relu_cache[l] = activation_out
        #print("shape of linear cache %d: " % l, linear_cache[l].shape)
        # activation_out = relu_forward(linear_cache[l])
        # relu_cache[l] = activation_out

        scores = linear_cache[l].copy() #activation_out.copy()
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores
        loss, grads = 0, dict()

        """
        TODO: Implement the backward pass for the fully-connected net. Store
        the loss in the loss variable and all gradients in the grads
        dictionary. Compute the loss with softmax. grads[k] has the gradients
        for self.params[k]. Add L2 regularisation to the loss function.
        NOTE: To ensure that your implementation matches ours and you pass the
        automated tests, make sure that your L2 regularization includes a
        factor of 0.5 to simplify the expression for the gradient.
        """
        #######################################################################
        #                           BEGIN OF YOUR CODE                        #
        #######################################################################
        # print(self.dropout_params)
        loss, dlogits = softmax(scores, y)
        #print("shape of dlogits: ", dlogits.shape)
        # add L2 regularization
        loss += 0.5 * self.reg * np.sum([np.sum(self.params["W%d" % l]**2)
                                          for l in range(1, self.num_layers + 1)])

        # last hidden layer <- output layer
        dX_, dW_, db_ = linear_backward(dlogits,
                                     relu_cache[self.num_layers-1],
                                     self.params["W%d"%(self.num_layers)],
                                     self.params["b%d" %(self.num_layers)])
        # add regularization effect to W
        grads["W%d"%(self.num_layers)] = dW_ + self.reg * self.params["W%d"%(self.num_layers)]
        grads["b%d" % (self.num_layers)] = db_
        #dX_ = relu_backward(dX_, )
        # hidden layer i <- hidden layer i+1
        for l in reversed(range(2, self.num_layers)):
            #print(l)
            # dX_ = dropout_backward(dX_, p=self.dropout_params["p"],
            #         train=self.dropout_params["train"], seed=self.dropout_params["seed"])
            if self.use_dropout:
                dX_ = dropout_backward(dX_, dropout_cache[l],
                        p=self.dropout_params["p"], train=self.dropout_params["train"])
            dX_ = relu_backward(dX_, linear_cache[l])
            dX_, dW_, db_ = linear_backward(dX_,
                                            relu_cache[l-1],
                                            self.params["W%d" % l],
                                            self.params["b%d" % l])
            # add regularization effect to W
            grads["W%d" % l] = dW_ + self.reg * self.params["W%d" % l]
            grads["b%d" % l] = db_
        # input layer <- first hidden layer
        if self.use_dropout:
            dX_ = dropout_backward(dX_, dropout_cache[1],
                    p=self.dropout_params["p"], train=self.dropout_params["train"])
        dX_ = relu_backward(dX_, linear_cache[1])
        dX_, dW_, db_ = linear_backward(dX_,
                                        X,
                                        self.params["W1"],
                                        self.params["b1"])
        # add regularization effect to W
        grads["W1"] = dW_ + self.reg * self.params["W1"]
        grads["b1"] = db_




        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
        return loss, grads
