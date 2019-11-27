from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import json
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from keras import backend

from python import imagenet
from python.slalom.models import get_model

def main(_):

    with tf.Graph().as_default():
        device = '/cpu:0'
        config = tf.ConfigProto(log_device_placement=False)
        config.allow_soft_placement = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.90
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            with tf.device(device):
                model, model_info = get_model('vgg_16', 1, include_top=True, double_prec=False)
                one_layer = backend.function([model.layers[7].input], [model.layers[7].output])
                images = tf.zeros((1, 224,224,3)).eval()
                inputs = tf.zeros((1, 56,56,128)).eval()
                #sess.run(model.outputs[0], feed_dict={model.inputs[0]:images, backend.learning_phase():0});
            for i in range(32):
                start_time = time.time()
                with tf.device(device):
                    one_layer([inputs])
                print("GPU compute time: {}".format(time.time() - start_time))

if __name__ == '__main__':
    tf.app.run()
