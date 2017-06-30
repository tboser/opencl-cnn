"""
Module that contains methods for forward passes of various layers found in a CNN.

Modules will initially be written in standard python and then have parts pushed onto the FPGA.
"""
from __future__ import division

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
    #take transpose to make operations simpler
    weights_transpose = filter_weights[0].T
    #print("weights_transpose:", weights_transpose)
    bias_vector = filter_weights[1]
    #print("bias_vector:", bias_vector)

    #pad matrix to maintain input dimensions
    filter_dim = len(weights_transpose[0][0])
    #print("filter_dim", filter_dim)
    if padding == 'same':
        num_zeros_padding = (filter_dim - 1) / 2
        padded_matrix = zero_pad_matrix(input_matrix, num_zeros_padding).T
    else:
        padded_matrix = input_matrix.T
    #print("padded matrix:", padded_matrix)
    #note: take transpose to make operations simpler

    #going to manually do the convolution to make conversion to opencl easier
    #this part will be done on an FPGA (until the return statement)
    #there are so many nested loops here something's got to change
    padded_matrix_dim = len(padded_matrix[0][0])
    #print("padded_matrix_dim:", padded_matrix_dim)
    num_filters = len(weights_transpose)
    #print("len padded_matrix:", len(padded_matrix))
    #print("len padded_matrix[0]:", len(padded_matrix[0]))
    #print("len padded_matrix[0][0]:", len(padded_matrix[0][0]))
    #print("len padded_matrix[0][0][0]:", len(padded_matrix[0][0][0]))
    #print("padded_matrix:", padded_matrix)
    #print("padded_matrix[0]:", padded_matrix[0])
    #print("padded_matrix[0][0]:", padded_matrix[0][0])
    #print("padded_matrix[0][0][0]:", padded_matrix[0][0][0])
    #this is shit has to be a better way
    if not isinstance(padded_matrix[0][0][0], np.float64) and not isinstance(padded_matrix[0][0][0], np.float32):
        num_input_channels = len(padded_matrix[0][0][0])
    else:
        num_input_channels = 1
    #num_input_channels = 1 # TEMP TODO
    output_matrix_width_height = padded_matrix_dim - filter_dim + 1
    output_matrix = np.ndarray(shape=(1, output_matrix_width_height,
                                      output_matrix_width_height, num_filters), dtype=np.float32)
    #print("len output_matrix:", len(output_matrix))
    #print("len output_matrix[0]:", len(output_matrix[0]))
    #print("len output_matrix[0][0]:", len(output_matrix[0][0]))
    #print("len output_matrix[0][0][0]:", len(output_matrix[0][0][0]))
    for j in range(0, padded_matrix_dim - filter_dim, stride): #height
        for i in range(0, padded_matrix_dim - filter_dim, stride): #width
            for k in range(0, num_filters): #filters
                weight_matrix = weights_transpose[k]
                conv_result = 0
                for h in range(0, num_input_channels): #depth
                    for l in range(0, filter_dim):
                        weight_vect = weight_matrix[h][l]
                        #if i == 0 and j == 0:
                        #    print("weight_vect:", weight_vect)
                        #    print("matrix it's being mult by:", padded_matrix[0][j+l][i:i+3])
                        if num_input_channels == 1:
                            conv_result += np.dot(weight_vect, padded_matrix[0][j+l][i:i+3])
                        else:
                            conv_result += np.dot(weight_vect, padded_matrix[0][j+l][i:i+3][h])
                #print(np.add(conv_result, bias_vector[k]))
                output_matrix[0][i][j][k] = np.add(conv_result, bias_vector[k])

    #print(output_matrix)
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
    for i in range(0, input_matrix_width_height-kernel_size):
        for j in range(0, input_matrix_width_height-kernel_size):
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
    input_matrix_depth = len(input_matrix[0][0][0])
    input_matrix_width_height = len(input_matrix[0])
    output_matrix_width_height = int(2 * input_matrix_width_height)

    output_matrix = np.ndarray(shape=(1, output_matrix_width_height, output_matrix_width_height, 
                                      input_matrix_depth), dtype=np.float32)

    for i in range(0, input_matrix_width_height):
        for j in range(0, input_matrix_width_height):
            for k in range(0, input_matrix_depth):
                output_matrix[0][i*2][j*2][k] = input_matrix[0][i][j][k]
                output_matrix[0][(i*2)+1][j*2][k] = input_matrix[0][i][j][k]
                output_matrix[0][i*2][(j*2)+1][k] = input_matrix[0][i][j][k]
                output_matrix[0][(i*2)+1][(j+2)+1][k] = input_matrix[0][i][j][k]
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
