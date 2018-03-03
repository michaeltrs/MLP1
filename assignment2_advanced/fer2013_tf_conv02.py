import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2
import pandas as pd
from scipy.ndimage import imread


def get_FER2013_data(datadir, labels, validation_ratio=None, mirror=True):
    train_id = np.array([True if file[:5] == 'Train' else False for file in labels['img']])
    
    print("Loading data...")

    xtrain = np.array([imread(datadir + '/' + name)[:, :, 0] for name in labels['img'][train_id]])
    # xtrain = np.array([Image.open(datadir + '/' + name) for name in labels['img'][train_id]])
    mean = xtrain.mean()
   
    xtrain = xtrain - mean
    ytrain = labels['emotion'][train_id].values

    print("Train data are of size %s" % str(xtrain[0].shape))

    xtest = np.array([imread(datadir + '/' + name)[:, :, 0] for name in labels['img'][~train_id]])
    # xtest = (xtest - mean) / np.sqrt(var)
    xtest = xtest - mean
    ytest = labels['emotion'][~train_id].values

    if validation_ratio:
        num_data = xtrain.shape[0]

        num_data_val = np.floor(validation_ratio * num_data).astype(int)
        val_data_indices = np.array([True]*num_data_val + [False]*(num_data - num_data_val))
        # np.random.shuffle(val_data_indices)

        xval = xtrain[val_data_indices]
        yval = ytrain[val_data_indices]

        xtrain = xtrain[~val_data_indices]
        ytrain = ytrain[~val_data_indices]
    else:
        xval = None
        yval = None
    
    if mirror:
        # xtrain_mirror = np.array(
        #     [np.fliplr(imread(datadir + '/' + name)[:, :, 0]) for name in labels['img'][train_id]])
        xtrain_mirror = np.fliplr(xtrain)
        xtrain = np.concatenate((xtrain, xtrain_mirror), axis=0)
        ytrain = np.concatenate((ytrain, ytrain))

    return xtrain, ytrain, xval, yval, xtest, ytest


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
num_epochs_no_improve = 50

datadir = ("/data/mat10/CO395/CW2/datasets/FER2013")
savedir = ("/data/mat10/CO395/CW2/conv02_logs")
model_name = "/conv02"
# USER INPUT - END ---------------------------------------------------- #

labels = pd.read_csv(datadir + '/labels_public.txt')

sess = tf.InteractiveSession()

# This flag is used to allow/prevent batch normalization params updates
# depending on whether the model is being trained or used for prediction.
training = tf.placeholder_with_default(True, shape=())

# Specify computation graph
xbatch = tf.placeholder(tf.float32, shape=[None, width, height, depth])
print("shape of input is %s" % xbatch.get_shape)

ybatch = tf.placeholder(tf.float32, shape=[None, num_labels])
print("shape of output is %s" % ybatch.get_shape)


# CNN model
conv1 = tf.layers.conv2d(inputs=xbatch,
                         filters=32,
                         kernel_size=[3, 3],
                         padding="same",
                         activation=tf.nn.relu)
print("shape of conv1 is %s" % conv1.get_shape)

conv2 = tf.layers.conv2d(inputs=conv1,
                         filters=32,
                         kernel_size=[3, 3],
                         padding="same",
                         activation=tf.nn.relu)
print("shape of conv2 is %s" % conv2.get_shape)

#conv3 = tf.layers.conv2d(inputs=conv2,
#                         filters=32,
#                         kernel_size=[3, 3],
#                         padding="same",
#                         activation=tf.nn.relu)
#print("shape of conv3 is %s" % conv3.get_shape)

pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
print("shape of pool1 is %s" % pool1.get_shape)


conv4 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
print("shape of conv4 is %s" % conv4.get_shape)

conv5 = tf.layers.conv2d(
      inputs=conv4,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
print("shape of conv5 is %s" % conv5.get_shape)

#conv6 = tf.layers.conv2d(
#      inputs=conv5,
#      filters=64,
#      kernel_size=[3, 3],
#      padding="same",
#      activation=tf.nn.relu)
#print("shape of conv6 is %s" % conv6.get_shape)


# Pooling Layer #2
# Second max pooling layer with a 2x2 filter and stride of 2
# Input Tensor Shape: [batch_size, 14, 14, 64]
# Output Tensor Shape: [batch_size, 7, 7, 64]
pool2 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)
print("shape of pool2 is %s" % pool2.get_shape)


conv7 = tf.layers.conv2d(
      inputs=pool2,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
print("shape of conv7 is %s" % conv7.get_shape)

conv8 = tf.layers.conv2d(
      inputs=conv7,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
print("shape of conv8 is %s" % conv8.get_shape)


pool3 = tf.layers.max_pooling2d(inputs=conv8, pool_size=[2, 2], strides=2)
print("shape of pool3 is %s" % pool3.get_shape)


# Flatten tensor into a batch of vectors
# Input Tensor Shape: [batch_size, 7, 7, 64]
# Output Tensor Shape: [batch_size, 7 * 7 * 64]
pool3_flat = tf.reshape(pool3, [-1, 6 * 6 * 128])
print("shape of pool3_flat is %s" % pool3_flat.get_shape)
dropout = tf.layers.dropout(
  inputs=pool3_flat, rate=0.4)

# Dense Layer
# Densely connected layer with 1024 neurons
# Input Tensor Shape: [batch_size, 7 * 7 * 64]
# Output Tensor Shape: [batch_size, 1024]
dense1 = tf.layers.dense(inputs=pool3_flat, units=512, activation=tf.nn.relu)
print("shape of dense1 is %s" % dense1.get_shape)

# Add dropout operation; 0.6 probability that element will be kept
dropout1 = tf.layers.dropout(
  inputs=dense1, rate=0.4)

dense2 = tf.layers.dense(inputs=dense1, units=128, activation=tf.nn.relu)
print("shape of dense2 is %s" % dense2.get_shape)

# Add dropout operation; 0.6 probability that element will be kept
dropout2 = tf.layers.dropout(
  inputs=dense2, rate=0.4)

# Logits layer
# Input Tensor Shape: [batch_size, 1024]
# Output Tensor Shape: [batch_size, 10]
logits = tf.layers.dense(inputs=dropout2, units=7)



# Train and Evaluate the Model
# set up for optimization (optimizer:ADAM)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ybatch, logits=logits))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)  # 1e-4
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(ybatch, 1))
train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




xtrain, ytrain, xval, yval, xtest, ytest = get_FER2013_data(datadir, labels, validation_ratio=0.05, mirror=True)

xtrain = xtrain.reshape(xtrain.shape[0], height, width, depth)
ytrain = make_one_hot(ytrain, 7)

xval = xval.reshape(xval.shape[0], height, width, depth)
yval = make_one_hot(yval, 7)

xtest = xtest.reshape(xtest.shape[0], height, width, depth)
ytest = make_one_hot(ytest, 7)

print("Training data shape is %s" % str(xtrain.shape))
print("Training data labels shape is %s" % str(ytrain.shape))
print("Validation data shape is %s" % str(xval.shape))
print("Validation data labels shape is %s" % str(yval.shape))
print("Test data shape is %s" % str(xtest.shape))
print("Test data labels shape is %s" % str(ytest.shape))

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())


train_accuracy_hist = []
val_accuracy_hist = []
best_val_accuracy = 0.
best_epoch = 0
# Include keep_prob in feed_dict to control dropout rate.
for epoch in range(num_epochs):

    print("epoch %d" % epoch)

    num_iter = xtrain.shape[0] // batch_size
    rem = xtrain.shape[0] - num_iter * batch_size
    for i in range(num_iter):

        xdata = xtrain[i*batch_size:(i+1)*batch_size]
        ydata = ytrain[i*batch_size:(i+1)*batch_size]
        train_step.run(feed_dict={xbatch: xdata, ybatch: ydata})

    # Logging every epoch
    pred = []
    for j in range(xtrain.shape[0] // batch_size):
        pred.append(logits.eval(feed_dict={xbatch: xtrain[j * batch_size:(j + 1) * batch_size]}))
    pred = np.concatenate(pred)
    ypred = np.argmax(pred, axis=1)
    train_accuracy = float((ypred == np.argmax(ytrain[:pred.shape[0]], axis=1)).sum()) / pred.shape[0]
    print("epoch %d, train accuracy %g" % (epoch, train_accuracy))
    train_accuracy_hist.append(train_accuracy)

    #if epoch % 1 == 0:
    pred = []
    for j in range(xval.shape[0] // batch_size):
        pred.append(logits.eval(feed_dict={xbatch: xval[j * batch_size:(j + 1) * batch_size]}))
    pred = np.concatenate(pred)
    ypred = np.argmax(pred, axis=1)
    val_accuracy = float((ypred == np.argmax(yval[:pred.shape[0]], axis=1)).sum()) / pred.shape[0]
    print("epoch %d, validation accuracy %g" % (epoch, val_accuracy))
    val_accuracy_hist.append(val_accuracy)

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_epoch = epoch
        saver.save(sess=sess, save_path=savedir + model_name)
        print("best model saved, validation accuracy is %.4f" % best_val_accuracy)

    if (epoch - best_epoch) > num_epochs_no_improve:
        print("No improvement found in a while, stopping optimization.")
        # Break out from the for-loop.
        break


np.savetxt(savedir + "/train_accuracy_history.csv", np.array(train_accuracy_hist))
np.savetxt(savedir + "/val_accuracy_history.csv", np.array(val_accuracy_hist))

