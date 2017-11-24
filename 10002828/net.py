import config
import os,random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import train_batch
import test_batch
import h5py
import sklearn
from sklearn.model_selection import KFold
from keras.utils import np_utils

# hyper parameters
np.random.seed(1337)  # for reproducibility
learning_rate = 0.001
batch_size = 128
training_epochs = 40
display_step = 1

# network hyper parameters 
n_input = 784
n_classes = 48
dropout = 0.7
dropout2 = 0.65

data_x = np.array(h5py.File("/home/gen/research/data/10002828kfold.h5",'r').get('data_x'))
label_y = np.array(h5py.File("/home/gen/research/data/10002828kfold.h5",'r').get('label_y'))

# define convolution
def conv2d(name, x, W, b, strides = 1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x, name=name)

# define pooling
def maxpool2d(name, x, k=2):
    return tf.nn.max_pool(x, ksize = [1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

# define normlize
def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

# network parameters
weights = {
    'wc1': tf.Variable(tf.random_normal([4, 4, 1, 96], stddev=0.1)),
    'wc2': tf.Variable(tf.random_normal([3, 3, 96, 256], stddev=0.1)),
    'wc3': tf.Variable(tf.random_normal([2, 2, 256, 384], stddev=0.1)),
    #'wc4': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=0.1)),
    #'wc5': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.1)),
    'wd1': tf.Variable(tf.random_normal([4*4*384, 4096], stddev=0.1)),
    #'wd2': tf.Variable(tf.random_normal([4096, 4096], stddev=0.1)),
    'out': tf.Variable(tf.random_normal([4096, n_classes], stddev=0.1))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([96], stddev=0.1)),
    'bc2': tf.Variable(tf.random_normal([256], stddev=0.1)),
    'bc3': tf.Variable(tf.random_normal([384], stddev=0.1)),
    #'bc4': tf.Variable(tf.random_normal([256], stddev=0.1)),
    #'bc5': tf.Variable(tf.random_normal([512], stddev=0.1)),
    'bd1': tf.Variable(tf.random_normal([4096], stddev=0.1)),
    #'bd2': tf.Variable(tf.random_normal([4096], stddev=0.1)),
    'out': tf.Variable(tf.random_normal([n_classes], stddev=0.1))
}


def net(x, weights, biases, dropout, dropout2):
    # reshape
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # conv1
    # conv
    conv1 = conv2d('conv1', x, weights['wc1'], biases['bc1'])
    # pooling
    pool1 = maxpool2d('pool1', conv1, k=2)
    drop1 = tf.nn.dropout(pool1, dropout)
    # normlize
    #norm1 = norm('norm1', drop1, lsize=4)

    # conv2
    # conv
    conv2 = conv2d('conv2', drop1, weights['wc2'], biases['bc2'])
    # pooling
    pool2 = maxpool2d('pool2', conv2, k=2)
    drop2 = tf.nn.dropout(pool2, dropout)
    # normlize
    #norm2 = norm('norm2', drop2, lsize=4)

    # conv3
    # conv
    conv3 = conv2d('conv3', drop2, weights['wc3'], biases['bc3'])
    #conv4 = conv2d('conv4', conv3, weights['wc4'], biases['bc4'])
    # pooling
    pool3 = maxpool2d('pool3', conv3, k=2)
    drop3 = tf.nn.dropout(pool3, dropout)

    # conv4
    #conv4 = conv2d('conv4', pool3, weights['wc4'], biases['bc4'])
    # pooling
    # pool4 = maxpool2d('pool3', conv4, k=2)
    # drop4 = tf.nn.dropout(pool4, dropout)
    # normlize
    # norm3 = norm('norm3', conv4, lsize=4)

    # conv4
    # conv4 = conv2d('conv4', norm3, weights['wc4'], biases['bc4'])
    # conv5
    #conv5 = conv2d('conv5', conv4, weights['wc5'], biases['bc5'])
    # pooling
    # pool4 = maxpool2d('pool4', conv5, k=2)
    # normlize
    # norm4 = norm('norm4', conv5, lsize=4)

    # fc1
    fc1 = tf.reshape(drop3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # dropout
    fc1 = tf.nn.dropout(fc1, dropout2)

    # fc2
    #fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
    #fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
    #fc2 = tf.nn.relu(fc2)

    # dropout
    #fc2 = tf.nn.dropout(fc2, dropout)

    # output
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# define placeholder
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout
keep_prob2 = tf.placeholder(tf.float32) #dropout2

# create model
pred = net(x, weights, biases, keep_prob, keep_prob2)

# define lost_func & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#score
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#define initialize
init = tf.global_variables_initializer()

#define k-fold cross validation
kf = KFold(n_splits=5, shuffle=False)

count = 0
for train_index, test_index in kf.split(data_x):
    print("%s, %s" % (train_index, test_index))
    img_rows,img_cols = config.img_row, config.img_col
    X_train, X_test = data_x[train_index], data_x[test_index]
    y_train, y_test = label_y[train_index], label_y[test_index]
    print(X_train.shape)
    print(X_test.shape)

    X_train = X_train.reshape(X_train.shape[0], img_rows * img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_rows * img_cols)

    print(X_train.shape)
    print(X_test.shape)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    print(y_train)
    print(y_test)

    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)

    print(y_train.shape)
    print(y_test)

    train = train_batch.DataSet(X_train, y_train)
    test = test_batch.DataSet(X_test, y_test)

    sess = tf.Session()
    sess.run(init)
    print ("GRAPH READY")

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(X_train.shape[0]/batch_size)
        total_batch_2 = int(X_test.shape[0]/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob:dropout, keep_prob2:dropout2})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob:1., keep_prob2:1.})/total_batch

        # Display logs per epoch step
        if epoch % display_step == 0:
            avg_train = 0
            for j in range(total_batch):
                batch_xs, batch_ys = train.next_batch(batch_size)
                train_acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob:1., keep_prob2:1.})
                avg_train += train_acc
            avg_train = avg_train / total_batch
            print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
            print (" Training accuracy: %.4f" % (avg_train))
            avg_test = 0
            for k in range(total_batch_2):
                batch_xt, batch_yt = test.next_batch(batch_size)
                test_acc = sess.run(accuracy, feed_dict={x: batch_xt, y: batch_yt, keep_prob:1., keep_prob2:1.})
                avg_test += test_acc
            avg_test = avg_test / total_batch_2
            print (" Test accuracy: %.4f" % (avg_test))
    count += 1
    print ("OPTIMIZATION FINISHED")
    print (count)
