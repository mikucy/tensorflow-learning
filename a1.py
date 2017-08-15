import tensorflow as tf


# 1b
# x = tf.random_uniform([], -1, 1)
# y = tf.random_uniform([], -1, 1)
# f1 = lambda: tf.add(x, y)
# f2 = lambda: tf.subtract(x, y)
# out = tf.case([(tf.less(x, y), f1)], default=f2)
#
# with tf.Session() as sess:
#     print("x,y,out=", sess.run([x, y, out]))

# 1c
# x = tf.constant([0, -2, -1, 0, 1, 2], dtype=tf.float32, shape=(2, 3))
# y = tf.zeros(x.shape)
# out = tf.equal(x, y)
#
# with tf.Session() as sess:
#     print("x,y,out=", sess.run([x, y, out]))

# 1d
# x = tf.constant([29.05088806,  27.61298943,  31.19073486,  29.35532951,
#                  30.97266006,  26.67541885,  38.08450317,  20.74983215,
#                  34.94445419,  34.45999146,  29.06485367,  36.01657104,
#                  27.88236427,  20.56035233,  30.20379066,  29.51215172,
#                  33.71149445,  28.59134293,  36.05556488,  28.66994858])
# out = tf.gather(x, tf.where(x > 30))
# with tf.Session() as sess:
#     print("out=", sess.run(out))

# 1e
# out = tf.diag(tf.range(1, 7, 1))
# with tf.Session() as sess:
#     print("out=", sess.run(out))

# 1f
# x = tf.random_uniform([10, 10])
# out = tf.matrix_determinant(x)
# with tf.Session() as sess:
#     print("x,out=", sess.run([x, out]))

# 1g
# x = tf.constant([5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9])
# out, idx = tf.unique(x)
# with tf.Session() as sess:
#     print("out=", sess.run(out))

# 1h
# x = tf.random_normal([300], mean=5, stddev=1)
# y = tf.random_normal([300], mean=5, stddev=1)
# average = tf.reduce_mean(x - y)
# def f1(): return tf.reduce_mean(tf.square(x - y))
# def f2(): return tf.reduce_sum(tf.abs(x - y))
# out = tf.cond(average < 0, f1, f2)
# with tf.Session() as sess:
#     print("out=", sess.run(out))

