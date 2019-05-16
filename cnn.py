import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

data_train = pd.read_csv('./class_f_train.csv', header = None)
data_test = pd.read_csv('./class_f_test.csv', header = None)



def getClass(x,type_):
    x_train = []
    #print(x.shape)
    for i in range(0,x.shape[0]):
        #print(x.iloc[i][0])
        #print('***********')
        imgFile = x.iloc[i][0][:8]
        im1 = cv2.imread("./compressedfft/"+type_+"/"+imgFile+".png",0)
        im1 = im1/255
        im1 = 1-im1
        #plt.imshow(im1)
        #plt.show()
        im1 = np.array(im1)
        #im1 = im1.ravel()
        #print(len(im1))
        x_train.append(im1)
    return x_train

def getLabel(y,type_):
    y_train = []
    for i in range(0,y.shape[0]):
        label = y.iloc[i][1]
        if(label == 'Acoustic_guitar'):
            #Acoustic_guitar.append(1)
            y_train.append(0)
        elif(label == 'Applause'):
            #Applause.append(1)
            y_train.append(1)
        elif(label == 'Bark'):
            #Bark.append(1)
            y_train.append(2)
        elif(label == 'Cough'):
            #Cough.append(1)
            y_train.append(3)
        elif(label == 'Flute'):
            #Flute.append(1)
            y_train.append(4)
        else:
            pass

       
    return y_train



x_train = getClass(data_train,"train")
x_train = np.array(x_train)

y_train = getLabel(data_train,"train")
y_train = np.array(y_train)

x_test = getClass(data_test,"test")
x_test = np.array(x_test)

y_test = getLabel(data_test,"test")
y_test = np.array(y_test)

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 100, 100, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 10 * 10 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=5)

    predictions = {
    # Generate predictions (for PREDICT and EVAL mode)
    "classes": tf.argmax(input=logits, axis=1),
    # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
    # `logging_hook`.
    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
    "accuracy": tf.metrics.accuracy(
    labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


mnist_classifier = tf.estimator.Estimator(
model_fn=cnn_model_fn, model_dir="./")

tensors_to_log = {"probabilities": "softmax_tensor"}

logging_hook = tf.train.LoggingTensorHook(
tensors=tensors_to_log, every_n_iter=50)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": x_train},
    y=y_train,
    batch_size=100,
    num_epochs=None,
    shuffle=True)


mnist_classifier.train(
    input_fn=train_input_fn,
    steps=1,
    hooks=[logging_hook])

print(x_train.shape)
