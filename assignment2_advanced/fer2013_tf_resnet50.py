import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2
import pandas as pd
from scipy.ndimage import imread


def get_FER2013_data(datadir, labels, validation_ratio=None, mirror=True):
    train_id = np.array([True if file[:5] == 'Train' else False for file in labels['img']])
    xtrain = np.array([imread(datadir + '/' + name)[:, :, 0] for name in labels['img'][train_id]])
    # xtrain = np.array([Image.open(datadir + '/' + name) for name in labels['img'][train_id]])
    mean = xtrain.mean()
    # var = xtrain.var(axis=0)
    # xtrain = (xtrain - mean) / np.sqrt(var)
    xtrain = xtrain - mean
    ytrain = labels['emotion'][train_id].values

    xtest = np.array([imread(datadir + '/' + name)[:, :, 0] for name in labels['img'][~train_id]])
    # xtest = (xtest - mean) / np.sqrt(var)
    xtest = xtest - mean
    ytest = labels['emotion'][~train_id].values

    if validation_ratio:
        num_data = xtrain.shape[0]

        num_data_val = np.floor(validation_ratio * num_data).astype(int)
        val_data_indices = np.array([True]*num_data_val + [False]*(num_data - num_data_val))
        np.random.shuffle(val_data_indices)

        xval = xtrain[val_data_indices]
        yval = ytrain[val_data_indices]

        xtrain = xtrain[~val_data_indices]
        ytrain = ytrain[~val_data_indices]
    else:
        xval = None
        yval = None

    if mirror:
        xtrain_mirror = np.array(
            [np.fliplr(imread(datadir + '/' + name)[:, :, 0]) for name in labels['img'][train_id]])
        xtrain = np.concatenate((xtrain, xtrain_mirror), axis=0)
        ytrain = np.concatenate((ytrain, ytrain))

    return xtrain, ytrain, xtest, ytest


def make_one_hot(y, num_labels):
    num_data = y.shape[0]
    yonehot = np.zeros((num_data, num_labels))
    yonehot[np.arange(num_data), y] = 1
    return yonehot


# USER INPUT - START ---------------------------------------------------#
width = 48
height = 48
depth = 1
batch_size = 128
num_labels = 7
num_epochs = 200
num_epochs_no_improve = 20

datadir = ("/data/mat10/CO395/CW2/datasets/FER2013")
savedir = ("/data/mat10/CO395/CW2/resnet50_logs")
# USER INPUT - END ---------------------------------------------------- #

labels = pd.read_csv(datadir + '/labels_public.txt')

sess = tf.InteractiveSession()
saver = tf.train.Saver()

# This flag is used to allow/prevent batch normalization params updates
# depending on whether the model is being trained or used for prediction.
training = tf.placeholder_with_default(True, shape=())

xbatch = tf.placeholder(tf.float32, shape=[None, width, height, depth])
# xval = tf.placeholder(tf.float32, shape=[batch_size, width, height, depth])

print("shape of input is %s" % xbatch.get_shape)

ybatch = tf.placeholder(tf.float32, shape=[None, num_labels])
print("shape of output is %s" % ybatch.get_shape)


# resnet model - need to change that to 34 layer model
resout, end_points = resnet_v2.resnet_v2_50(xbatch, num_classes=num_labels)
resout = tf.reshape(resout, (batch_size, 7))
print("shape after resnet is %s" % resout.get_shape)

# ypred = tf.nn.softmax(resout)
# print("shape after softmax is %s" % ypred.get_shape)

# Train and Evaluate the Model
# set up for optimization (optimizer:ADAM)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ybatch, logits=resout))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)  # 1e-4
correct_prediction = tf.equal(tf.argmax(resout, 1), tf.argmax(ybatch, 1))
train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


xtrain, ytrain, xtest, ytest = get_FER2013_data(datadir, labels, mirror=False)
xtrain = xtrain.reshape(xtrain.shape[0], height, width, depth)
ytrain = make_one_hot(ytrain, 7)
xtest = xtest.reshape(xtest.shape[0], height, width, depth)
ytest = make_one_hot(ytest, 7)


sess.run(tf.global_variables_initializer())


train_accuracy_hist = []
test_accuracy_hist = []
best_test_accuracy = 0.
best_epoch = 0
# Include keep_prob in feed_dict to control dropout rate.
for epoch in range(num_epochs):

    print("epoch %d of %d" % (epoch, num_epochs))

    num_iter = xtrain.shape[0] // batch_size
    rem = xtrain.shape[0] - num_iter * batch_size
    for i in range(num_iter):

        xdata = xtrain[i*batch_size:(i+1)*batch_size]
        ydata = ytrain[i*batch_size:(i+1)*batch_size]
        train_step.run(feed_dict={xbatch: xdata, ybatch: ydata})

        # if i % 500 == 0:
        #     # print("iteration step %d" % i)
        #     ypred = resout.eval(feed_dict={xbatch: xdata})
        #     ypred = np.argmax(ypred, axis=1)
        #     train_accuracy = (ypred == np.argmax(ydata[:ydata.shape[0]], axis=1)).sum() / ypred.shape[0]
        #     print("epoch %d, iteration step %d, train accuracy %g" % (epoch, i, train_accuracy))

    # # Logging every 100th iteration in the training process.
    # if epoch % 1 == 0:
    #     pred = []
    #     for j in range(xtest.shape[0] // batch_size):
    #         pred.append(resout.eval(feed_dict={xbatch: xtest[j * batch_size:(j + 1) * batch_size]}))
    #
    #     pred = np.concatenate(pred)
    #     ypred = np.argmax(pred, axis=1)
    #     test_accuracy = (ypred == np.argmax(ytest[:pred.shape[0]], axis=1)).sum() / pred.shape[0]
    #
    #     print("epoch %d, test accuracy %g" % (epoch, test_accuracy))
    # Logging every epoch
    pred = []
    for j in range(xtrain.shape[0] // batch_size):
        pred.append(resout.eval(feed_dict={xbatch: xtrain[j * batch_size:(j + 1) * batch_size]}))
    pred = np.concatenate(pred)
    ypred = np.argmax(pred, axis=1)
    train_accuracy = (ypred == np.argmax(ytrain[:pred.shape[0]], axis=1)).sum() / pred.shape[0]
    print("epoch %d, train accuracy %g" % (epoch, train_accuracy))
    train_accuracy_hist.append(train_accuracy)

    pred = []
    for j in range(xtest.shape[0] // batch_size):
        pred.append(resout.eval(feed_dict={xbatch: xtest[j * batch_size:(j + 1) * batch_size]}))
    pred = np.concatenate(pred)
    ypred = np.argmax(pred, axis=1)
    test_accuracy = (ypred == np.argmax(ytest[:pred.shape[0]], axis=1)).sum() / pred.shape[0]
    print("epoch %d, test accuracy %g" % (epoch, test_accuracy))
    test_accuracy_hist.append(test_accuracy)

    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        best_epoch = epoch
        saver.save(sess=sess, save_path=savedir)

    if (epoch - best_epoch) > num_epochs_no_improve:
        print("No improvement found in a while, stopping optimization.")
        # Break out from the for-loop.
        break

np.savetxt(savedir + "/train_accuracy.csv", np.array(train_accuracy_hist))
np.savetxt(savedir + "/test_accuracy.csv", np.array(test_accuracy_hist))
# ypred = resout.eval(feed_dict={xbatch: xdata})
# ypred = np.argmax(ypred, axis=1)
# train_accuracy = (ypred == np.argmax(ydata[:ydata.shape[0]], axis=1)).sum() / ypred.shape[0]
# a = x_input[0,0,:, :, 0]#[0]
# b = a[:, :][0]
# np.savetxt("/homes/mat10/Programming/OpenCV/frames/test.csv", a)
# a = test_var1[0][0][0]
# test_var1[:, 0, :, :, :]
