# coding=UTF-8 

from flask import Flask, request, jsonify, abort, make_response
from redis import Redis, RedisError
import os
import socket

from PIL import Image, ImageFilter

#from cassandra.cluster import Cluster
#from cassandra import ConsistencyLevel
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import logging
import time

log = logging.getLogger()
log.setLevel('INFO')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
log.addHandler(handler)

# Connect to Redis
redis = Redis(host="redis", db=0, socket_connect_timeout=2, socket_timeout=2)

app = Flask(__name__)
KEYSPACE = "mnist_data1"

#Get image 
def getimage():
    file_name = './new.png' 
    im = Image.open(file_name).convert('L')
    im.save("./sample.png")
    # plt.imshow(im)
    # plt.show()
    tv = list(im.getdata()) 
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


#Prediction
def Prediction():
    result = getimage()
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    x_image = tf.reshape(x, [-1, 28, 28, 1])
  # First convolutional layer - maps one grayscale image to 32 feature maps.
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

  #load the model
    saver = tf.train.Saver()
    with tf.Session() as sess:
        #sess.run(init_op)
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, "./model.ckpt")
        # print ("Model restored.")

        prediction = tf.argmax(y_conv, 1)
        predict = prediction.eval(feed_dict={x: [result], keep_prob: 1.0}, session=sess)
        print(h_conv2)

        # print('recognize result:')
        # print(predint[0])
        return str(predict[0])

def insert_data(filename, result, req_time):
    cluster = Cluster(contact_points=['127.0.0.1'], port=9042)
    session = cluster.connect()
    
    try:
        session.execute("""
        CREATE KEYSPACE %s
        WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '2' }
        """ % KEYSPACE)
        log.info("Setting keyspace")
        session.set_keyspace(KEYSPACE)
        session.execute("""
           CREATE TABLE mytable (
           filename text,
           result text,
           time text,
           PRIMARY KEY (filename, time)
           )
           """)
    except Exception as error:
        log.error("Unable to create table")
        log.error(error)

    log.info("Setting keyspace")
    session.set_keyspace(KEYSPACE)

    log.info("Starting keyspace...")
    try:
        log.info("inserting table...")
        session.execute("""
           INSERT INTO mytable (filename, result, time)
           VALUES ('%s', '%s', '%s')
           """ % (filename, result, req_time))
    except Exception as e:
        log.error("Unable to insert data")
        log.error(e)


@app.route('/mnist', methods=['GET', 'POST'])
def upload_file():
  if request.method == 'POST':
    f = request.files['file']
    f.save('./new.png')
    upload_filename = f.filename
    result = Prediction()
    req_time = str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    insert_data(upload_filename, result, req_time)
  return "%s%s%s%s%s%s%s%s%s" % ("Upload File Name: ", upload_filename, "\n",
                                   "Result: ", result, "\n",
                                   "Upload Time: ", req_time, "\n")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7000)

