#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Authors:
# Matvey Ezhov / https://github.com/ematvey/tensorflow-seq2seq-tutorials
# Mathias Müller / mmueller@cl.uzh.ch

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import helpers
import constants as C

class Model():

    def __init__(self,
                 vocab_size=10,
                 input_embedding_size=20,
                 encoder_hidden_units=20,
                 decoder_hidden_units=20):

        self.vocab_size = vocab_size
        self.input_embedding_size = input_embedding_size
        self.encoder_hidden_units = encoder_hidden_units
        self.decoder_hidden_units = decoder_hidden_units

    def build_embeddings(self):

        self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
        self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

        self.decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')

        self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.input_embedding_size], -1.0, 1.0), dtype=tf.float32)

        self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)
        self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)

    def build_encoder(self):

        self.encoder_cell = tf.contrib.rnn.LSTMCell(self.encoder_hidden_units)

        self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(
            self.encoder_cell, self.encoder_inputs_embedded,
            dtype=tf.float32, time_major=True,
        )

    def build_decoder(self):

        self.decoder_cell = tf.contrib.rnn.LSTMCell(self.decoder_hidden_units)

        self.decoder_outputs, self.decoder_final_state = tf.nn.dynamic_rnn(
            self.decoder_cell, self.decoder_inputs_embedded,

            initial_state=self.encoder_final_state,

            dtype=tf.float32, time_major=True, scope="plain_decoder",
        )

    def build(self):
        """
        Build a tensorflow graph.
        """
        self.build_embeddings()
        self.build_encoder()
        self.build_decoder()

        # finally a linear layer on top of the decoder outputs
        self.decoder_logits = tf.contrib.layers.linear(self.decoder_outputs, self.vocab_size)

        self.decoder_prediction = tf.argmax(self.decoder_logits, 2)

        self.stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.one_hot(self.decoder_targets, depth=self.vocab_size, dtype=tf.float32),
            logits=self.decoder_logits,
        )

        self.loss = tf.reduce_mean(self.stepwise_cross_entropy)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        self.init = tf.global_variables_initializer()

    def forward(self, batch_):
        """
        Forward a batch through the current model.
        """
        with tf.Session() as sess:
            sess.run(m.init)

            batch_, batch_length_ = helpers.batch(batch_)
            print('batch_encoded:\n' + str(batch_))

            din_, dlen_ = helpers.batch(np.ones(shape=(3, 1), dtype=np.int32),
                                        max_sequence_length=4)
            print('decoder inputs:\n' + str(din_))

            pred_ = sess.run(self.decoder_prediction,
                feed_dict={
                    self.encoder_inputs: batch_,
                    self.decoder_inputs: din_,
                })
            print('decoder predictions:\n' + str(pred_))

    def _next_feed(self, batches):
        batch = next(batches)
        encoder_inputs_, _ = helpers.batch(batch)
        decoder_targets_, _ = helpers.batch(
            [(sequence) + [C.EOS] for sequence in batch]
        )
        decoder_inputs_, _ = helpers.batch(
            [[C.EOS] + (sequence) for sequence in batch]
        )
        return {
            self.encoder_inputs: encoder_inputs_,
            self.decoder_inputs: decoder_inputs_,
            self.decoder_targets: decoder_targets_,
        }

    def train(self, batch_size=100, max_batches=1000):

        with tf.Session() as sess:

            sess.run(self.init)

            batches = helpers.random_sequences(length_from=3, length_to=8,
                                           vocab_lower=2, vocab_upper=10,
                                           batch_size=batch_size)

            print('head of the batch:')
            for seq in next(batches)[:10]:
                print(seq)

            self.loss_track = []
            batches_in_epoch = 100


            for batch in range(max_batches):
                fd = self._next_feed(batches)
                _, l = sess.run([self.train_op, self.loss], fd)
                self.loss_track.append(l)

                if batch == 0 or batch % batches_in_epoch == 0:
                    print('batch {}'.format(batch))
                    print('  minibatch loss: {}'.format(sess.run(self.loss, fd)))
                    predict_ = sess.run(self.decoder_prediction, fd)
                    for i, (inp, pred) in enumerate(zip(fd[self.encoder_inputs].T, predict_.T)):
                        print('  sample {}:'.format(i + 1))
                        print('    input     > {}'.format(inp))
                        print('    predicted > {}'.format(pred))
                        if i >= 2:
                            break
                    print()


# create model instance
m = Model()

# build model
m.build()

# try current predictions with a dummy batch
batch_ = [[6], [3, 4], [9, 8, 7]]
m.forward(batch_)

# train model
m.train(batch_size=100, max_batches=1000)
