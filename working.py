import numpy as np
from utils.data_utils import *

datadir = ('/home/mat10/Documents/MSc Machine Learning/395-Machine Learning/'
           'CW2/assignment2_advanced/datasets/cifar-10-batches-py/')

data = load_CIFAR_batch(datadir + 'data_batch_1')


X = data[0][0:10]
X.shape

N = 10
M = 100
D = np.prod(X[0].shape)

W = np.random.randn(D, M)
b = np.random.randn(M)

out = linear_forward(X, W, b)
out.shape

#X = out.copy()
dout = np.random.randn(N, M)
dout.shape

dX, dW, db = linear_backward(dout, X, W, b)

