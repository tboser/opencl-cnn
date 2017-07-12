"""
Tests for functions in forward_pass.py
"""

from __future__ import print_function

from keras import layers, models
from keras import backend as K
from termcolor import colored

from data_utils import get_layer_type
from forward_pass import convolution_2d, max_pool_2d, upsampling_2d, reshape, sigmoid, relu

import numpy as np

def compare_ndarrays(keras_ndarray, comparison_ndarray):
    """
    Returns total difference between two ndarrays.
    """
    if keras_ndarray.shape != comparison_ndarray.shape:
        print("Unable to compare ndarrays with different shape.")
        print("Keras ndarray shape:", keras_ndarray.shape)
        print("Your ndarray shape:", comparison_ndarray.shape)
        return -1

    if np.array_equal(keras_ndarray, comparison_ndarray):
        print("Keras ndarray and comparison ndarray are the same.")
        return 0

    print("Keras ndarray and comparison ndarray are NOT the same.")
    print("A total of", np.sum(keras_ndarray == comparison_ndarray)," out of",
                               keras_ndarray.size ," elements are the same.")
    return abs(np.sum(keras_ndarray - comparison_ndarray))

def keras_get_layer_output(model, layer, test_input):
    """
    Helper method, gives the output matrix from a Keras layer
    """
    get_layer_output = K.function([model.layers[0].input],
                                  [layer.output])
    return get_layer_output([test_input])[0]

def predict_with_keras(model, layer_number, tinput):
    """
    To test other functions
    """
    get_layer_output = K.function([model.layers[layer_number-1].input],
                                   [model.layers[layer_number].output])
    return get_layer_output([tinput][0])

def full_run_comparison(keras_model, test_input):
    """
    From start to finish compare the output of a Keras prediction to
    a prediction done using forward_pass.py
    """
    previous_layer_output_keras = test_input
    previous_layer_output_self = test_input

    for layer in keras_model.layers:

        #get layer information via keras
        layer_type = get_layer_type(layer)
        layer_config = layer.get_config()

        #perform appropriate operation
        if layer_type == 'convolution2d':
            previous_layer_output_self = convolution_2d(previous_layer_output_self, layer.get_weights(), 1, padding='same')
            self_out_with_keras_input = convolution_2d(previous_layer_output_keras, layer.get_weights(), 1, padding='same')
            if layer_config['activation'] == 'relu':
                previous_layer_output_self = relu(previous_layer_output_self)
                self_out_with_keras_input = relu(self_out_with_keras_input)
            if layer_config['activation'] == 'sigmoid':
                previous_layer_output_self = sigmoid(previous_layer_output_self)
                self_out_with_keras_input = sigmoid(self_out_with_keras_input)
        elif layer_type == 'maxpooling2d':
            previous_layer_output_self = max_pool_2d(previous_layer_output_self, layer_config['pool_size'][0], layer_config['strides'][0], padding='valid')
            self_out_with_keras_input = max_pool_2d(previous_layer_output_keras, layer_config['pool_size'][0], layer_config['strides'][0], padding='valid')
        elif layer_type == 'reshape':
            previous_layer_output_self = reshape(previous_layer_output_self, layer_config['target_shape'])
            self_out_with_keras_input = reshape(previous_layer_output_keras, layer_config['target_shape'])
        elif layer_type == 'upsampling2d':
            previous_layer_output_self = upsampling_2d(previous_layer_output_self, layer_config['size'][0])
            self_out_with_keras_input = upsampling_2d(previous_layer_output_keras, layer_config['size'][0])
        else:
            continue

        #perform keras prediction for this layer
        previous_layer_output_keras = keras_get_layer_output(keras_model, layer, test_input)

        #now let's see how we compare to keras
        print("\n===========================================================================")
        print("Forward pass on a", layer_config['name'],"layer.")
        if np.array_equal(self_out_with_keras_input, previous_layer_output_keras):
            print(colored("PASS: This layer successfully produces the same results as Keras.", 'green'))
        else:
            print(colored("FAIL: This layer does no perform in the same way as Keras.", 'red'))
        if previous_layer_output_keras.shape != self_out_with_keras_input.shape:
            print(colored("CRITICAL ERROR: This layer returns the wrong output shape.", 'red'))
        print("Using self output as input", np.sum(previous_layer_output_self == previous_layer_output_keras),
                                                   " of", previous_layer_output_keras.size, " correctly predicted")
        print("Using keras output as input", np.sum(self_out_with_keras_input == previous_layer_output_keras),
                                                    " of", previous_layer_output_keras.size, " correctly predicted")
        if np.allclose(self_out_with_keras_input, previous_layer_output_keras, atol=0.00001):
            print(colored("SUCCESS: All output values within 0.00001 of eachother.", 'green'))
        else:
            print(colored("FAIL: All output values NOT within 0.00001 of eachother.", 'red'))
        print("===========================================================================")

