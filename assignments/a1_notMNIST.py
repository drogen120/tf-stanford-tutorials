import numpy as np
from sklearn.cross_validation import train_test_split
from PIL import Image
import os

class NotMNIST:
    def __init__(self):
        images, labels = [], []

        for i, letter in enumerate(['A','B','C','D','E','F','G','H','I','J']):
            directory = '../data/notMNIST_small/%s/' % letter
            files = os.listdir(directory)
            label = np.array([0]*10)
            label[i] = 1
            for file in files:
                try:
                    im = Image.open(directory+file)
                except:
                    print "Skip a corrupted file:" + file
                    continue
                pixels = np.array(im.convert('L').getdata())
                images.append(pixels/255.0)
                labels.append(label)

        train_images, test_images, train_labels, test_labels = \
            train_test_split(images, labels, test_size=0.2, random_state=0)

        train_images = np.asarray(train_images)
        test_images = np.asarray(test_images)
        train_labels = np.asarray(train_labels)
        test_labels = np.asarray(test_labels)

        class train:
            def __init__(self):
                self.images = []
                self.labels = []
                self.batch_counter = 0

            def next_batch(self, num):
                if self.batch_counter + num >= len(self.labels):
                    batch_images = self.images[self.batch_counter:,:]
                    batch_labels = self.labels[self.batch_counter:,:]
                    left = num - len(batch_labels)
                    #batch_images.append(self.images[:left,:])
                    #batch_labels.append(self.images[:left,:])
                    batch_images = np.concatenate((batch_images, self.images[:left,:]), axis = 0)
                    batch_labels = np.concatenate((batch_labels, self.labels[:left,:]), axis = 0)
                    self.batch_counter = left
                else:
                    batch_images = self.images[self.batch_counter:self.batch_counter + num,:]
                    batch_labels = self.labels[self.batch_counter:self.batch_counter + num,:]
                    self.batch_counter += num

                return (batch_images, batch_labels)

        class test:
            def __init__(self):
                self.images = []
                self.labels = []
                self.test_counter = 0

            def next_batch(self, num):
                if self.test_counter + num >= len(self.labels):
                    batch_images = self.images[self.test_counter:,:]
                    batch_labels = self.labels[self.test_counter:,:]
                    left = num - len(batch_labels)
                    #batch_images.append(self.images[:left,:])
                    #batch_labels.append(self.images[:left,:])
                    batch_images = np.concatenate((batch_images, self.images[:left,:]), axis = 0)
                    batch_labels = np.concatenate((batch_labels, self.labels[:left,:]), axis = 0)
                    self.test_counter += num
                else:
                    batch_images = self.images[self.test_counter:self.test_counter + num,:]
                    batch_labels = self.labels[self.test_counter:self.test_counter + num,:]
                    self.test_counter += num

                return (batch_images, batch_labels)

        self.train = train()
        self.test = test()

        self.train.images = train_images
        self.train.labels = train_labels
        self.test.images = test_images
        self.test.labels = test_labels

import matplotlib.pyplot as plt
import tensorflow as tf
nomnist = NotMNIST()

learning_rate = 0.01
batch_size = 128
n_epochs = 200

X = tf.placeholder(tf.float32, [batch_size, 784])
Y = tf.placeholder(tf.float32, [batch_size, 10])

w1 = tf.Variable(tf.random_normal(shape=[784,400], stddev = 0.01), name = "weights_1")
b1 = tf.Variable(tf.zeros([1, 400]), name = "bias_1")

w2 = tf.Variable(tf.random_normal(shape = [400, 200], stddev = 0.01), name = "weights_2")
b2 = tf.Variable(tf.zeros([1, 200]), name = "bias_2")

w3 = tf.Variable(tf.random_normal(shape = [200, 10], stddev = 0.01), name = "weights_3")
b3 = tf.Variable(tf.zeros([1, 10]), name = "bias_3")

layer1 = tf.matmul(X, w1) + b1
re1 = tf.nn.relu(layer1)

layer2 = tf.matmul(re1, w2) + b2
re2 = tf.nn.relu(layer2)

logits = tf.matmul(re2, w3) + b3
entropy = tf.nn.softmax_cross_entropy_with_logits(logits, Y)
loss = tf.reduce_mean(entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    summary_writer = tf.summary.FileWriter("./graph", graph = sess.graph)
    tf.summary.scalar("loss", loss)

    summary_op = tf.summary.merge_all()
    n_batches = int(len(nomnist.train.labels) / batch_size)
    print n_batches
    k = 0
    for i in range(n_epochs):
        for _ in range(n_batches):
            X_datas, Y_datas = nomnist.train.next_batch(batch_size)
            #print X_datas[0].shape
            _, loss_p = sess.run([optimizer, loss], feed_dict = {X : X_datas, Y : Y_datas})
            print loss_p
            summary_str = sess.run(summary_op, feed_dict = {X : X_datas, Y : Y_datas})
            summary_writer.add_summary(summary_str, k)
            k = k + 1

    total_correct_preds = 0
    n_batches = int(len(nomnist.test.labels) / batch_size)
    for i in range(n_batches):
        X_tests, Y_tests = nomnist.test.next_batch(batch_size)
        _, loss_batch, logits_batch = sess.run([optimizer, loss ,logits], feed_dict = {X : X_tests, Y : Y_tests})
        preds = tf.nn.softmax(logits_batch)
        correct_preds = tf.equal(tf.argmax(preds,1), tf.argmax(Y_tests,1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
        total_correct_preds += sess.run(accuracy)

    print "Accuracy {0}".format(total_correct_preds/nomnist.test.test_counter)




# fig = plt.figure(figsize=(8,8))
# for i in range(10):
#     c = 0
#     for (image, label) in zip(nomnist.test.images, nomnist.test.labels):
#         if np.argmax(label) != i: continue
#         subplot = fig.add_subplot(10, 10, i*10+c+1)
#         subplot.set_xticks([])
#         subplot.set_yticks([])
#         subplot.imshow(image.reshape((28,28)), vmin=0, vmax = 1, \
#             cmap=plt.cm.gray_r, interpolation="nearest")
#         c += 1
#         if c == 10: break
# plt.show()
