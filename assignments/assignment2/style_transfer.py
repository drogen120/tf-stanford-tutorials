""" An implementation of the paper "A Neural Algorithm of Artistic Style"
by Gatys et al. in TensorFlow.

Author: Chip Huyen (huyenn@stanford.edu)
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
For more details, please read the assignment handout:
http://web.stanford.edu/class/cs20si/assignments/a2.pdf
"""
from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf

import vgg_model
import utils

# parameters to manage experiments
STYLE = 'starry_night'
CONTENT = 'deadpool'
STYLE_IMAGE = 'styles/' + STYLE + '.jpg'
CONTENT_IMAGE = 'content/' + CONTENT + '.jpg'
IMAGE_HEIGHT = 250
IMAGE_WIDTH = 333
NOISE_RATIO = 0.6 # percentage of weight of the noise for intermixing with the content image

# Layers used for style features. You can change this.
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
W = [0.5, 1.0, 1.5, 1.5, 3.0] # give more weights to deeper layers.

# Layer used for content features. You can change this.
CONTENT_LAYER = 'conv4_2'

ITERS = 3000
LR = 0.005

MEAN_PIXELS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
""" MEAN_PIXELS is defined according to description on their github:
https://gist.github.com/ksimonyan/211839e770f7b538e2d8
'In the paper, the model is denoted as the configuration D trained with scale jittering. 
The input images should be zero-centered by mean pixel (rather than mean image) subtraction. 
Namely, the following BGR values should be subtracted: [103.939, 116.779, 123.68].'
"""

# VGG-19 parameters file
VGG_DOWNLOAD_LINK = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
EXPECTED_BYTES = 534904783

def _create_content_loss(p, f):
    """ Calculate the loss between the feature representation of the
    content image and the generated image.
    
    Inputs: 
        p, f are just P, F in the paper 
        (read the assignment handout if you're confused)
        Note: we won't use the coefficient 0.5 as defined in the paper
        but the coefficient as defined in the assignment handout.
    Output:
        the content loss

    """
    half_value = tf.constant(1/(75*4.0), dtype=tf.float32)
    result = tf.reduce_sum(tf.square(tf.subtract(p, f)))
    return tf.multiply(half_value, result)

def _gram_matrix(F, N, M):
    """ Create and return the gram matrix for tensor F
        Hint: you'll first have to reshape F
    """
    #F = tf.reshape(F, [-1])
    print (N,M)
    #N = tf.constant(N, dtype=tf.int32)
    #M = tf.constant(M, dtype=tf.int32)
    f = tf.reshape(F, [-1, M, N], name="g_reshape")
    result = tf.matmul(f, f, transpose_a=True, transpose_b=False)
    return result

def _single_style_loss(a, g):
    """ Calculate the style loss at a certain layer
    Inputs:
        a is the feature representation of the real image
        g is the feature representation of the generated image
    Output:
        the style loss at a certain layer (which is E_l in the paper)

    Hint: 1. you'll have to use the function _gram_matrix()
        2. we'll use the same coefficient for style loss as in the paper
        3. a and g are feature representation, not gram matrices
    """
    a = tf.constant(a, dtype=tf.float32)
    [_, m1, m2, n] = g.get_shape()

    m = int(m1) * int(m2)
    n = int(n)
    ag = _gram_matrix(a, n, m)
    gg = _gram_matrix(g, n, m)
    e_sum = tf.reduce_sum(tf.square(tf.subtract(ag, gg)))
    c1 = tf.constant(2, dtype=tf.float32)
    mn = tf.constant(m*n, dtype=tf.float32)
    e = tf.div(e_sum, tf.square(tf.multiply(c1, mn)))
    return e

def _create_style_loss(A, model):
    """ Return the total style loss
    """
    n_layers = len(STYLE_LAYERS)
    E = [_single_style_loss(A[i], model[STYLE_LAYERS[i]]) for i in range(n_layers)]
    
    ###############################
    ## TO DO: return total style loss
    #wl = tf.constant(0.2, dtype=tf.float32)
    wl = tf.constant(W, dtype=tf.float32)
    #sum_e = tf.multiply(tf.reduce_sum(E), wl)
    sum_e = tf.reduce_sum(tf.multiply(E, wl))

    # sum_e = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    # for i in range(n_layers):
    #     sum_e = tf.add(sum_e, tf.multiply(wl, E[i]))
    return sum_e
    ###############################

def _create_losses(model, input_image, content_image, style_image):
    with tf.variable_scope('loss') as scope:
        with tf.Session() as sess:
            sess.run(input_image.assign(content_image)) # assign content image to the input variable
            p = sess.run(model[CONTENT_LAYER])
        content_loss = _create_content_loss(p, model[CONTENT_LAYER])

        with tf.Session() as sess:
            sess.run(input_image.assign(style_image))
            A = sess.run([model[layer_name] for layer_name in STYLE_LAYERS])                              
        style_loss = _create_style_loss(A, model)

        ##########################################
        ## TO DO: create total loss. 
        ## Hint: don't forget the content loss and style loss weights
        a = tf.constant(0.0001, dtype=tf.float32)
        total_loss = tf.add(tf.multiply(a, content_loss), style_loss)
        ##########################################

    return content_loss, style_loss, total_loss

def _create_summary(model):
    """ Create summary ops necessary
        Hint: don't forget to merge them
    """
    tf.summary.scalar('content_loss', model['content_loss'])
    tf.summary.scalar('texture_loss', model['style_loss'])
    tf.summary.scalar('total_loss', model['total_loss'])
    summary_op = tf.summary.merge_all()
    return summary_op

def train(model, generated_image, initial_image):
    """ Train your model.
    Don't forget to create folders for checkpoints and outputs.
    """
    skip_step = 1
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(init)
        ###############################
        ## TO DO: 
        ## 1. initialize your variables
        ## 2. create writer to write your graph
        ###############################
        sess.run(generated_image.assign(initial_image))
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        initial_step = model['global_step'].eval()
        writer = tf.summary.FileWriter(logdir='graph/mylog', graph=sess.graph)
        start_time = time.time()
        for index in range(initial_step, ITERS):
            if index >= 5 and index < 20:
                skip_step = 10
            elif index >= 20:
                skip_step = 20
            
            sess.run(model['optimizer'])
            if (index + 1) % skip_step == 0:
                ###############################
                ## TO DO: obtain generated image and loss
                total_loss, summary, gen_image = sess.run([model['total_loss'], model['summary_op'], generated_image])
                ###############################
                gen_image = gen_image + MEAN_PIXELS
                writer.add_summary(summary, global_step=index)
                print('Step {}\n   Sum: {:5.1f}'.format(index + 1, np.sum(gen_image)))
                print('   Loss: {:5.1f}'.format(total_loss))
                print('   Time: {}'.format(time.time() - start_time))
                start_time = time.time()

                filename = 'outputs/%d.png' % (index)
                utils.save_image(filename, gen_image)

                if (index + 1) % 20 == 0:
                    saver.save(sess, 'checkpoints/style_transfer', index)

def main():
    with tf.variable_scope('input') as scope:
        # use variable instead of placeholder because we're training the intial image to make it
        # look like both the content image and the style image
        input_image = tf.Variable(np.zeros([1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]), dtype=tf.float32)
    
    utils.download(VGG_DOWNLOAD_LINK, VGG_MODEL, EXPECTED_BYTES)
    model = vgg_model.load_vgg(VGG_MODEL, input_image)
    model['global_step'] = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    
    content_image = utils.get_resized_image(CONTENT_IMAGE, IMAGE_HEIGHT, IMAGE_WIDTH)
    content_image = content_image - MEAN_PIXELS
    style_image = utils.get_resized_image(STYLE_IMAGE, IMAGE_HEIGHT, IMAGE_WIDTH)
    style_image = style_image - MEAN_PIXELS

    model['content_loss'], model['style_loss'], model['total_loss'] = _create_losses(model, 
                                                    input_image, content_image, style_image)
    ###############################
    ## TO DO: create optimizer
    model['optimizer'] = tf.train.GradientDescentOptimizer(LR).minimize(model['total_loss'], global_step=model['global_step'])
    ###############################
    model['summary_op'] = _create_summary(model)

    initial_image = utils.generate_noise_image(content_image, IMAGE_HEIGHT, IMAGE_WIDTH, NOISE_RATIO)
    train(model, input_image, initial_image)

if __name__ == '__main__':
    main()
