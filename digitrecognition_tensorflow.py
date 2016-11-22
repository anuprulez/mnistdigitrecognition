import pandas as pd
from pandas import DataFrame
import tensorflow as tf
# numpy
import numpy as np
import datetime

# get digit train and test files
print "loading files..."
a = datetime.datetime.now()
digit_train = pd.read_csv("train.csv")
digit_test = pd.read_csv("test.csv")
b = datetime.datetime.now()
print (b-a)
print "loading files finished"

digit_train_label = digit_train["label"]
digit_train = digit_train.drop("label", axis=1)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

train_data_conv = []

def get_data_batch(starting_index, batch_size, train_data):
    set_train = []
    set_label = []
    for i in range( starting_index, starting_index + batch_size ):
        set_train.append( train_data[i] )
        set_label.append( y_values[i] )
    return [ set_train, set_label ]

def convert_data(data_):
    data_conv = []
    for i in range(0, len( data_ )):
        pixels_row = DataFrame(data=data_, index=[i] )
        row_data = list(pixels_row.values)
        data_conv.append(row_data[0])
    return data_conv

# Format label collection
y_values = np.zeros((len(digit_train_label), 10))

for i, item in enumerate(digit_train_label):
    y_values[i][item] = 1

x = tf.placeholder(tf.float32, shape=[None, 784]) #train data
y_ = tf.placeholder(tf.float32, shape=[None, 10]) #train labels

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# Prediction function
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Predicted label
prediction = tf.argmax(y_conv,1)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

print "training data conversion started..."
a = datetime.datetime.now()
train_data_conv = convert_data(digit_train)
b = datetime.datetime.now()
print (b-a)
print "training data conversion finished"

print "training in batches started..."
a = datetime.datetime.now()
iteration = 200
batch_size = 25
validation_set_size = 100
digit_train_length = len(train_data_conv) - validation_set_size
for i in range(iteration):
    batch_index = 0
    for j in range(0, digit_train_length/batch_size):
        batch = get_data_batch(batch_index, batch_size, train_data_conv)
        batch_index = batch_index + batch_size
        if batch_index % 10000 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0}, session=sess)
            print("iteration %d, step %d, training accuracy %g" % (i, batch_index, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}, session=sess)

b = datetime.datetime.now()
print (b-a)
print "training finished"

# Validation
a = datetime.datetime.now()
print "validation started..."
batch_validation = get_data_batch(len(train_data_conv) - validation_set_size, validation_set_size, train_data_conv)
validation_accuracy = accuracy.eval(feed_dict={x:batch_validation[0], y_: batch_validation[1], keep_prob: 1.0}, session=sess)
print("Validation accuracy %g"%(validation_accuracy)) 
print "validation finished"
b = datetime.datetime.now()
print (b-a)

# Test data conversion
a = datetime.datetime.now()
print "test data conversion started..."
test_data_conv = convert_data(digit_test)
b = datetime.datetime.now()
print (b-a)

# Prediction
a = datetime.datetime.now()
print "prediction started..."
test_batch_size = 100
test_batch_index = 0
test_prediction = []
for i in range(0, len(test_data_conv)/test_batch_size):
    test_batch = []
    for j in range(test_batch_index, test_batch_index + test_batch_size):
        test_batch.append(test_data_conv[j])
    test_batch_index = test_batch_index + test_batch_size
    # test batch data
    value = prediction.eval(feed_dict={x:test_batch, keep_prob: 1.0}, session=sess)
    for item in value:
        test_prediction.append(item)

# Save prediction result
image_id = []
for i in range(0, len(test_prediction)):
    image_id.append(i+1)

submit_results_tf = pd.DataFrame({
    "ImageId": image_id,
    "Label": test_prediction
})

submit_results_tf.to_csv('submission_TensorFlow.csv', index=False)

b = datetime.datetime.now()
print (b-a)
print "prediction finished"

b = datetime.datetime.now()
print "program ended at: "
print b
sess.close()
