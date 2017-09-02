from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from process_data import process_data

VOCAB_SIZE = 1000
BATCH_SIZE = 128
EMBED_SIZE = 128  # dimension of the word embedding vectors
SKIP_WINDOW = 1  # the context window
NUM_SAMPLED = 64  # Number of negative examples to sample.
LEARNING_RATE = 0.02
NUM_TRAIN_STEPS = 20000
SKIP_STEP = 2000  # how many steps to skip before reporting the loss
USE_UPDATE_IN_PLACE_INSTEAD_OF_MATRIX_ADD = False
WEIGHTS_FLD = 'processed/'


def save_embed_matrix(sess, embed_matrix):
    # code to visualize the embeddings. uncomment the below to visualize embeddings
    final_embed_matrix = sess.run(embed_matrix)
    # it has to variable. constants don't work here. you can't reuse model.embed_matrix
    embedding_var = tf.Variable(final_embed_matrix[:1000], name='embedding')
    sess.run(embedding_var.initializer)
    config = projector.ProjectorConfig()
    summary_writer = tf.summary.FileWriter(WEIGHTS_FLD)
    # add embedding to the config file
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    # link this tensor to its metadata file, in this case the first 500 words of vocab
    embedding.metadata_path = WEIGHTS_FLD + 'vocab_1000.tsv'
    # saves a configuration file that TensorBoard will read during startup.
    projector.visualize_embeddings(summary_writer, config)
    saver_embed = tf.train.Saver([embedding_var])
    # 单独保存包含词向量的Session数据为Checkpoint
    saver_embed.save(sess, WEIGHTS_FLD + 'model3.ckpt', 1)


def count(batch_gen, num_train_steps):
    with tf.Session() as sess:
        # build co-occurrence matrix
        co_occurrence_matrix = tf.Variable(tf.zeros([VOCAB_SIZE, VOCAB_SIZE], dtype=tf.int32), name="co_occurrence_matrix")
        # initialize
        sess.run(tf.global_variables_initializer())

        # method 1 update in place
        if USE_UPDATE_IN_PLACE_INSTEAD_OF_MATRIX_ADD:
            for step in range(num_train_steps):
                centers, targets = next(batch_gen)
                for index in range(BATCH_SIZE):
                    # find the position
                    x = int(centers[index])
                    y = int(targets[index][0])
                    # get the row
                    row = tf.gather(co_occurrence_matrix, x)
                    # update the row
                    new_row = tf.concat([row[:y], [co_occurrence_matrix[x][y]+1], row[y+1:]], axis=0)
                    # update the matrix
                    co_occurrence_matrix.assign(tf.scatter_update(co_occurrence_matrix, x, new_row))
                if (step + 1) % SKIP_STEP == 0:
                    do_svd(co_occurrence_matrix, sess, step)
        else:
            for step in range(num_train_steps):
                centers, targets = next(batch_gen)
                # find the position
                for index in range(BATCH_SIZE):
                    pos = [int(centers[index]), int(targets[index][0])]
                    # build a sparse matrix
                    pos_matrix = tf.sparse_tensor_to_dense(tf.SparseTensor([pos], [1], [VOCAB_SIZE, VOCAB_SIZE]))
                    co_occurrence_matrix.assign_add(pos_matrix)
                if (step + 1) % SKIP_STEP == 0:
                    do_svd(co_occurrence_matrix, sess, step)


def do_svd(co_occurrence_matrix, sess, step):
    # do svd to decompose the matrix
    s, u, v = tf.svd(tf.cast(co_occurrence_matrix, tf.float32))
    # decrease dimension to embed_size
    embed_matrix = tf.matmul(u[:, :EMBED_SIZE], tf.diag(s[:EMBED_SIZE]))
    # save the matrix
    save_embed_matrix(sess, embed_matrix)
    print('%s => Step %s' % (datetime.now(), step))


def main():
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    count(batch_gen, NUM_TRAIN_STEPS)

main()
