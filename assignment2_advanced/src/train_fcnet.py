import numpy as np

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data

"""
TODO: Use a Solver instance to train a TwoLayerNet that achieves at least 50% 
accuracy on the validation set.
"""
###########################################################################
#                           BEGIN OF YOUR CODE                            #
###########################################################################
datapath = datadir = ('/home/mat10/Documents/MSc Machine Learning/395-Machine Learning/'
           'CW2/assignment2_advanced/datasets/cifar-10-batches-py')
data = get_CIFAR10_data(datapath)

hidden_dims = [512, 256]
net = FullyConnectedNet(hidden_dims, num_classes=10, dropout=0., reg=0., seed=0)

solver = Solver(net,
                data,
                update_rule='sgd',
                optim_config={'learning_rate': 1e-3,
                              'momentum': 0.3},
                lr_decay=0.95,
                num_epochs=10,
                batch_size=100,
                print_every=10000)

solver.train()

make_plots = False
if make_plots:
    import matplotlib.pyplot as plt
    # declare model and solver and train the model
    # model = [...]
    # solver = [...]
    # solver.train()
    # Run this cell to visualize training loss and train / val accuracy
    plt.subplot(2, 1, 1)
    plt.title('Training loss')
    plt.plot(solver.loss_history, 'o')
    plt.xlabel('Iteration')
    plt.subplot(2, 1, 2)
    plt.title('Accuracy')
    plt.plot(solver.train_acc_history, '-o', label='train')
    plt.plot(solver.val_acc_history, '-o', label='val')
    plt.plot([0.5] * len(solver.val_acc_history), 'k--')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.gcf().set_size_inches(15, 12)
    plt.show()

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
