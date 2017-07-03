"""
Module that contains methods for forward passes of various layers found in a CNN.

Modules will initially be written in standard python and then have parts pushed onto the FPGA.
"""
from __future__ import print_function, division
from data_utils import get_layer_type
from tests import predict_with_keras, keras_get_layer_output

import numpy as np
import pyopencl as cl


def convolution_2d(input_matrix, filter_weights, stride, padding='same'):
    """
    Notes: filter_weights[0] is the weights, filter_weights[1] is the bias.

    This is not for training, this method takes a trained weight matrix as input.

    input_matrix: input features from previous layer (or initial input)
    filter_weights: trained weight matrix for this layer
    stride: stride of convolution
    padding: 'same' to pad for output with same dimensions, 'valid' for no padding

    Output:
    Matrix of values***todo***
    """
    weights_matrix = filter_weights[0]
    bias_vector = filter_weights[1]

    #pad matrix to maintain input dimensions
    filter_dim = len(weights_matrix)
    if padding == 'same':
        num_zeros_padding = (filter_dim - 1) / 2
        padded_matrix = zero_pad_matrix(input_matrix, num_zeros_padding)
    else:
        padded_matrix = input_matrix

    #going to manually do the convolution to make conversion to opencl easier
    #this part will be done on an FPGA (until the return statement)
    #there are so many nested loops here something's got to change
    padded_matrix_dim = len(padded_matrix[0][0])
    num_filters = len(weights_matrix.T)

    try:
        num_input_channels = len(input_matrix[0][0][0])
    except:
        num_input_channels = 1
    output_matrix_width_height = padded_matrix_dim - filter_dim + 1
    output_matrix = np.ndarray(shape=(1, output_matrix_width_height,
                                      output_matrix_width_height, num_filters), dtype=np.float32)

    for j in range(0, padded_matrix_dim - filter_dim + 1, stride): #height
        for i in range(0, padded_matrix_dim - filter_dim + 1, stride): #width
            for k in range(0, num_filters): #filters
                conv_result = 0
                for h in range(0, num_input_channels): #depth
                    for l in range(0, filter_dim):
                        if num_input_channels == 1:
                            for s in range(0, filter_dim):
                                conv_result += weights_matrix[l][s][0][k]*padded_matrix[0][i+l][j+s]
                        else:
                            for s in range(0, filter_dim):
                                conv_result += weights_matrix[l][s][h][k]*padded_matrix[0][i+l][j+s][h]
                output_matrix[0][i][j][k] = np.add(conv_result, bias_vector[k])

    return output_matrix


def max_pool_2d(input_matrix, kernel_size, stride, padding='valid'):
    """
    Simple max_pool layer.

    input_matrix:
    kernel_size: size of pooling kernel.
    stride: stride of pool

    For now I will assume padding is always valid
    MAJOR TODO:
    Currently assumes kernel size is 2x2 and stride is 2,2. 
    Need to figure out a function to go from input matrix indices to output matrix indices!
    """
    input_matrix_width_height = len(input_matrix[0])
    input_matrix_depth = len(input_matrix[0][0][0])
    #print("input_matrix_depth:", input_matrix_depth)
    output_matrix_width_height = int(((input_matrix_width_height - kernel_size) / stride) + 1)
    output_matrix = np.ndarray(shape=(1, output_matrix_width_height, output_matrix_width_height,
                                      input_matrix_depth), dtype=np.float32)
    #print("input matrix dimensions", input_matrix.shape)
    #print("output matrix dimensions:", output_matrix.shape)
    for i in range(0, input_matrix_width_height-kernel_size+1, stride):
        for j in range(0, input_matrix_width_height-kernel_size+1, stride):
            for k in range(0, input_matrix_depth):
                #ok now need to get max out of this array chunk, will have to revisit this.
                curr_max = 0
                for a in range(0, kernel_size):
                    for b in range(0, kernel_size):
                        curr_max = max(curr_max, input_matrix[0][i+a][j+b][k])
                output_matrix[0][int(i/2)][int(j/2)][k] = curr_max

    return output_matrix

def relu(input_matrix):
    """
    Relu activation function. f(x) = max(x, 0)

    TODO: 
    Do I want to implement this directly into the convolution_2d method? (same number of comparisions)
    Is there a more efficient way of doing this?
    """
    #iterate over every item in matrix, take max
    for i in range(0, len(input_matrix[0])):
        for j in range(0, len(input_matrix[0][0])):
            for k in range(0, len(input_matrix[0][0][0])):
                input_matrix[0][i][j][k] = max(0, input_matrix[0][i][j][k])

    return input_matrix

def sigmoid(input_matrix):
    """
    Sigmoid activation function
    S(x) = 1 / 1 + e^-x
    """
    for i in range(0, len(input_matrix[0])):
        for j in range(0, len(input_matrix[0][0])):
            for k in range(0, len(input_matrix[0][0][0])):
                input_matrix[0][i][j][k] = np.float32(1 / (1 + (np.exp(-input_matrix[0][i][j][k]))))

    return input_matrix

def upsampling_2d(input_matrix, kernel_size):
    """
    Increase width and height of input matrix.

    TODO - figure out how upsampling2d is supposed to work
         - figure out how to get output dimensions
    """
    #input_matrix_depth = len(input_matrix[0][0][0])
    #input_matrix_width_height = len(input_matrix[0])
    #output_matrix_width_height = int(2 * input_matrix_width_height)

    #output_matrix = np.ndarray(shape=(1, output_matrix_width_height, output_matrix_width_height, 
    #                                  input_matrix_depth), dtype=np.float32)

    #for i in range(0, input_matrix_width_height):
    #    for j in range(0, input_matrix_width_height):
    #        for k in range(0, input_matrix_depth):
    #            output_matrix[0][i*2][j*2][k] = input_matrix[0][i][j][k]
    #            output_matrix[0][(i*2)+1][j*2][k] = input_matrix[0][i][j][k]
    #            output_matrix[0][i*2][(j*2)+1][k] = input_matrix[0][i][j][k]
    #            output_matrix[0][(i*2)+1][(j+2)+1][k] = input_matrix[0][i][j][k]
    output_matrix = np.repeat(np.repeat(input_matrix, kernel_size, axis=1), kernel_size, axis=2)
    return output_matrix

def reshape(input_matrix, reshape_dims):
    """
    Reshape input hopefully the same way Keras does it.
    """
    return np.reshape(input_matrix, (1,) + reshape_dims)

def zero_pad_matrix(input_matrix, num_zeros):
    """
    Pad the 3d (nxmxz) input matrix with p zeros
    Assumes 'reshaped' matrix. Need to look at this more closely.
    """
    num_zeros = int(num_zeros)
    return np.pad(input_matrix, ((0, 0), (num_zeros, num_zeros),
                                 (num_zeros, num_zeros), (0, 0)), 'constant')

def predict_with_keras_model(keras_model, event):
    prediction = event
    for layer in keras_model.layers:

        #get layer information via keras
        layer_type = get_layer_type(layer)
        layer_config = layer.get_config()

        #perform appropriate operation
        if layer_type == 'convolution2d':
            prediction = convolution_2d(prediction, layer.get_weights(), 1, padding='same')
            if layer_config['activation'] == 'relu':
                prediction = relu(prediction)
            if layer_config['activation'] == 'sigmoid':
                prediction = sigmoid(prediction)
        if layer_type == 'maxpooling2d':
            prediction = max_pool_2d(prediction, layer_config['pool_size'][0], layer_config['strides'][0], padding='valid')
        if layer_type == 'reshape':
            prediction = reshape(prediction, layer_config['target_shape'])
        if layer_type == 'upsampling2d':
            #prediction = keras_get_layer_output(keras_model, layer, event)
            #prediction = predict_with_keras(keras_model, i, prediction)
            #temporarily use keras for this
            prediction = upsampling_2d(prediction, layer_config['size'][0])
    return prediction

