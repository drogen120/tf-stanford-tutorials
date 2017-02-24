from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os

class SkipGramModel:

    def __init__(self):

        self.VOCAB_SIZE = 50000
        self.BATCH_SIZE = 128
        self.EMBED_SIZE = 128  # dimension of the word embedding vectors
        self.SKIP_WINDOW = 1  # the context window
        self.NUM_SAMPLED = 64  # Number of negative examples to sample.
        self.LEARNING_RATE = 1.0
        self.NUM_TRAIN_STEPS = 10000
        self.SKIP_STEP = 2000  # how many steps to skip before reporting the loss
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        self._create_placeholders()
        self._create_embedding()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()

    def _create_placeholders(self):

        with tf.name_scope('data'):
            self.center_words = tf.placeholder(tf.int32, shape=[self.BATCH_SIZE], name = "center_words")
            self.target_words = tf.placeholder(tf.int32, shape=[self.BATCH_SIZE, 1], name = "target_words")

    def _create_embedding(self):

        with tf.name_scope('embedding'):
            self.embed_matrix = tf.Variable(tf.random_uniform([self.VOCAB_SIZE, self.EMBED_SIZE], minval=-1, maxval=1), name="embed_matrix")

    def _create_loss(self):

        with tf.name_scope('loss'):
            embed = tf.nn.embedding_lookup(self.embed_matrix, self.center_words, name='embed')
            nce_weights = tf.Variable(tf.truncated_normal([self.VOCAB_SIZE, self.EMBED_SIZE], stddev=1.0 / (self.EMBED_SIZE ** 0.5)))
            nce_bias = tf.Variable(tf.zeros([self.VOCAB_SIZE]))
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights, biases=nce_bias, labels=self.target_words, inputs=embed,
                               num_sampled=self.NUM_SAMPLED, num_classes=self.VOCAB_SIZE))

    def _create_optimizer(self):
        self.optimizer = tf.train.GradientDescentOptimizer(self.LEARNING_RATE).minimize(self.loss, global_step = self.global_step)

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram_loss", self.loss)
            self.summary_op = tf.summary.merge_all()

    def train(self, batch_gen):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # TO DO: initialize variables
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname("checkpoints/checkpoint"))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            total_loss = 0.0  # we use this to calculate the average loss in the last SKIP_STEP steps
            writer = tf.summary.FileWriter('./my_graph/good_vis/', sess.graph)
            init_step = self.global_step.eval()
            for index in xrange(init_step, init_step+self.NUM_TRAIN_STEPS):
                centers, targets = batch_gen.next()
                # TO DO: create feed_dict, run optimizer, fetch loss_batch
                _, loss_batch, summary = sess.run([self.optimizer, self.loss, self.summary_op], feed_dict={self.center_words: centers, self.target_words: targets})
                writer.add_summary(summary, index)
                total_loss += loss_batch
                if (index + 1) % self.SKIP_STEP == 0:
                    print('Average loss at step {}: {:5.1f}'.format(index, total_loss / self.SKIP_STEP))
                    total_loss = 0.0
                    saver.save(sess, "checkpoints/skipmodel", self.global_step)

            final_embed_matrix = sess.run(self.embed_matrix)
            embedding_var = tf.Variable(final_embed_matrix[:1000], name='embedding')
            sess.run(embedding_var.initializer)
            config = projector.ProjectorConfig()
            summary_writer = tf.summary.FileWriter('processed')
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name
            embedding.metadata_path = 'processed/vocab_1000.tsv'
            projector.visualize_embeddings(summary_writer, config)
            saver_embed = tf.train.Saver([embedding_var])
            saver_embed.save(sess, './my_graph/good_vis/model3.ckpt', 1)

            writer.close()
