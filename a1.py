import tensorflow as tf

x = tf.random_uniform([], -1, 1)
y = tf.random_uniform([], -1, 1)
f1 = lambda: tf.add(x, y)
f2 = lambda: tf.subtract(x, y)
out = tf.case([(tf.less(x, y), f1)], default=f2)

with tf.Session() as sess:
    print("x,y,out=", sess.run([x, y, out]))

