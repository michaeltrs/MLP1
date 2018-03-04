import numpy as np
import tensorflow as tf
from fer2013_tf_conv03 import *
import os
from scipy.ndimage import imread


def test_deep_fer_model(img_folder, model):
    """
    Given a folder with images, load the images (in lexico-graphical ordering
    according to the file name of the images) and your best model to predict
    the facial expression of each image.
    Args:
    - img_folder: Path to the images to be tested
    Returns:
    - preds: A numpy vector of size N with N being the number of images in
    img_folder.
    """
    ### Start your code here
    files = [file for file in os.listdir(img_folder) if file[-3:] == 'jpg']
    files = sorted(files, key=str.lower)
    test_data = np.array([imread(img_folder + '/' + file)[:, :, 0] for file in files])
    N, W, H = test_data.shape
    test_data = test_data.reshape(N, W, H, 1)

    sess = tf.InteractiveSession()
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    saver.restore(sess, model)  # "/tmp/model.ckpt")
    print("Model restored.")
    pred = logits.eval(feed_dict={xbatch: test_data})
    ypred = np.argmax(pred, axis=1)
    ### End of code
    return ypred


if __name__ == "__main__":

    # USER INPUT - START -------------------------------------------------- #
    # specify the data directory
    img_folder = ("/data/mat10/CO395/CW2/datasets/FER2013/Test")

    # specify the model directory
    modeldir = ("/data/mat10/CO395/CW2/conv03_logs")
    # USER INPUT - END ---------------------------------------------------- #

    model = modeldir + "/conv03"
    # make predictions using path to test data
    ypred = test_deep_fer_model(img_folder, model)
    print(ypred)
