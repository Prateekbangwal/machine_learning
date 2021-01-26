"""Importing the required libaries"""
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
from tensorflow.examples.tutorials.mnist import input_data
tf.logging.set_verbosity(tf.logging.ERROR)
import matplotlib.pyplot as plt

"""Loading the dataset"""

mnist = input_data.read_data_sets("data/mnist", one_hot=True)
print(mnist)


"""chechking the data"""
print("NO of images in training set{}".format(mnist.train.images.shape))
print("NO of Labels in training set{}".format(mnist.train.labels.shape))
print("NO of images in test set{}".format(mnist.test.images.shape))
print("NO of Labels in test set{}".format(mnist.test.labels.shape))

"""Defining the number of neurons in each layer"""

#number of neuron in input layer

num_input = 784

#number of neuron in hidden layer1

num_hidden1 = 512

#number of neuron in hidden layer2

num_hidden2 = 256
#number of neuron in hidden layer3

num_hidden3 = 128

#number of neuron in output layer

num_output = 10
# defining the name scopes

with tf.name_scope('input'):
    X = tf.placeholder('float', [None, num_input])
with tf.name_scope('output'):
    Y = tf.placeholder('float', [None, num_output])
with tf.name_scope('weights'):
    weights = {
        'w1': tf.Variable(tf.truncated_normal([num_input, num_hidden1], stddev=0.1), name='weight_1'),
        'w2': tf.Variable(tf.truncated_normal([num_hidden1, num_hidden2], stddev = 0.1), name='weight_2'),
        'w3': tf.Variable(tf.truncated_normal([num_hidden2, num_hidden3], stddev=0.1), name='weight_3'),
        'out': tf.Variable(tf.truncated_normal([num_hidden3, num_output], stddev=0.1), name='weight_4'),
    }
with tf.name_scope('biases'):
    biases = {
        'b1': tf.Variable(tf.constant(0.1, shape=[num_hidden1]), name='bias_1'),
        'b2': tf.Variable(tf.constant(0.1, shape=[num_hidden2]), name='bias_2'),
        'b3': tf.Variable(tf.constant(0.1, shape=[num_hidden3]), name='bias_3'),
        'out': tf.Variable(tf.constant(0.1, shape=[num_output]), name='bias_4')
    }

#Forward Propagation

with tf.name_scope('Model'):
    with tf.name_scope('layer1'):
        layer_1 = tf.nn.relu(tf.add(tf.matmul(X, weights['w1']), biases['b1']))
    with tf.name_scope('layer2'):
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['w2']), biases['b2']))
    with tf.name_scope('layer3'):
        layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['w3']), biases['b3']))
    with tf.name_scope('output_layer'):
        y_hat = tf.nn.sigmoid(tf.matmul(layer_3, weights['out'])+biases['out'])

# computing loss

with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=Y))

#back propagation
learning_rate = 0.0001
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Accuracy

with tf.name_scope("Accuracy"):
    predicted_digit = tf.argmax(y_hat, 1)
    actual_digit = tf.argmax(Y,1)
    correct_pred = tf.equal(predicted_digit, actual_digit)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.summary.scalar("Accuracy", accuracy)
tf.summary.scalar("Loss", loss)
merge_summary = tf.summary.merge_all()


init = tf.global_variables_initializer()
learning_rate = 0.0001
num_iterations = 1000
batch_size = 128

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter('./graphs', graph = sess.graph)

    for i in range(num_iterations):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={X:batch_x, Y:batch_y})
        if i % 100 == 0:
            batch_loss, batch_accuracy, summary = sess.run([loss, accuracy, merge_summary], feed_dict={X: batch_x, Y:batch_y})
            summary_writer.add_summary(summary, i)
            print('Iterations: {}, Loss:{}, Accuracy:{}'.format(i, batch_loss, batch_accuracy))