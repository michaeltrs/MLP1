import numpy as np
from src.utils.data_utils import *
from src.utils.solver import *
from src.fcnet import *

datadir = ('/home/mat10/Documents/MSc Machine Learning/395-Machine Learning/'
           'CW2/assignment2_advanced/datasets/cifar-10-batches-py/')

traindata = load_CIFAR_batch(datadir + 'data_batch_1')
testdata = load_CIFAR_batch(datadir + 'test_batch')

# X = data[0][0:10]
# X.shape
#
# N = 10
# M = 100
# D = np.prod(X[0].shape)
#
# W = np.random.randn(D, M)
# b = np.random.randn(M)
#
# out = linear_forward(X, W, b)
# out.shape
#
# #X = out.copy()
# dout = np.random.randn(N, M)
# dout.shape
#
# dX, dW, db = linear_backward(dout, X, W, b)

hidden_dims = [1024, 512]

N = 50

net = FullyConnectedNet(hidden_dims, num_classes=10,
                 dropout=0., reg=0.0)
#
# X = data[0][0:N]
# scores = net.loss(X)
#
# y = np.random.choice(10, N)
#
# loss, grads = net.loss(X, y)

data = {
      'X_train': traindata[0][:N],
      'y_train': traindata[1][:N],
      'X_val': testdata[0],
      'y_val': testdata[1]
       }

solver = Solver(net,
                data,
                update_rule='sgd',
                optim_config={'learning_rate': 1e-3},
                lr_decay=0.95,
                num_epochs=20,
                batch_size=10,
                print_every=100)
solver.train()




















#
#
# loss, dlogits = softmax(logits, y)
# dlogits.shape
#
#
#
#
#
# y = np.random.choice(10, N)
# loss, dlogits = softmax(scores, y)
# dlogits.shape
#
# net.params["W%d"%(net.num_layers)].shape
#
#
#
#
# net.params['W1'].shape
# net.params['b1'].shape
#
# net.num_layers
# net.params
#
#
#
#
# L2_loss = loss + 0.5 * net.reg * np.sum([np.sum(net.params["W%d"%l])
#                                          for l in range(1, net.num_layers + 1)])
