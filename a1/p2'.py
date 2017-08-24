import numpy as np
import tensorflow as tf

# Load the data
data = './data/heart.txt'
with open(data) as f:
    dataset = []
    line = f.readline()
    while line:
        line = f.readline()
        temp = line.split('\t', 10)
        if temp[0]:
            if temp[4] == "Present":
                temp[4] = 1
            else:
                temp[4] = 0
            for i in range(9):
                temp[i] = float(temp[i])
            if temp[9][0] == '1':
                temp[9] = 1
            else:
                temp[9] = 0
            dataset.append(temp)
    dataset = np.asarray(dataset)

# Shuffle and split the dataset into train and test part
dataset = tf.random_shuffle(dataset)
train_dataset = dataset[:400, :]
test_dataset = dataset[400:, :]

X = tf.cast(train_dataset[:, 0:9], tf.float32)
Y = tf.cast(train_dataset[:, 9:10], tf.float32)

w = tf.Variable(tf.random_normal(shape=[9, 1], stddev=0.01))
b = tf.Variable(tf.zeros([1, 1]))

logits = tf.matmul(X, w) + b
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y)
loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        _, total_loss = sess.run([optimizer, loss])
        print('Average loss epoch{0}: {1}'.format(i, total_loss))

    # Test the model.
    X_test = tf.cast(test_dataset[:, 0:9], tf.float32)
    Y_test = tf.cast(test_dataset[:, 9:10], tf.float32)
    inferenced = tf.sigmoid(tf.matmul(X_test, w) + b)
    preds = tf.cast(inferenced > 0.5, tf.float32)
    print(sess.run(tf.reduce_mean(tf.cast(tf.equal(preds, Y_test), tf.float32))))


