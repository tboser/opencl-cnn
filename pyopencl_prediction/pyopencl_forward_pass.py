from __future__ import print_function, division
from naive_prediction.data_utils import get_layer_type
#from tests import predict_with_keras, keras_get_layer_output

import numpy as np
import pyopencl as cl

def predict(model, test_events, prg, queue):
    """ make full prediction given test_events """
    all_predictions = []
    for event in test_events:
        #print(event.shape)
        event = np.reshape(event, event.shape + (1,))
        prediction = event
        for layer in model.layers:

            #get layer information via keras
            layer_type = get_layer_type(layer)
            layer_config = layer.get_config()

            #perform appropriate operation
            if layer_type == 'convolution2d':
                prediction = convolution_2d(prediction, layer.get_weights(), 1, prg, queue, activation=layer_config['activation'], padding='same')
            if layer_type == 'maxpooling2d':
                prediction = maxpooling_2d(prediction, layer_config['pool_size'][0], layer_config['strides'][0], prg, queue)
            #not parallelized
            if layer_type == 'reshape':
                prediction = reshape(prediction, layer_config['target_shape'])
            if layer_type == 'upsampling2d':
                prediction = upsampling_2d(prediction, layer_config['size'][0])
        all_predictions.append(prediction)
    return all_predictions

def convolution_2d(input_matrix, filter_weights, stride, prg, queue, activation='relu', padding='same'):

    #print(input_matrix)
    #print(input_matrix.shape)
    print(input_matrix.shape)
    weights_matrix = filter_weights[0]
    bias_vector = filter_weights[1]

    #pad matrix to maintain input dimensions
    filter_dim = len(weights_matrix)
    if padding == 'same':
        num_zeros_padding = (filter_dim - 1) / 2
        padded_matrix = zero_pad_matrix(input_matrix, num_zeros_padding)
    else:
        padded_matrix = input_matrix
        
    padded_matrix_dim = len(padded_matrix[0][0])
    num_filters = len(weights_matrix.T)

    try:
        num_input_channels = len(input_matrix[0][0][0])
    except:
        num_input_channels = 1
    output_matrix_width_height = padded_matrix_dim - filter_dim + 1
    
    a = cl.array.to_device(queue, padded_matrix)
    b = cl.array.to_device(queue, weights_matrix)
    c = cl.array.zeros(queue, shape=(1, output_matrix_width_height,
                                     output_matrix_width_height, num_filters),
                      dtype=np.float32)
    d = cl.array.to_device(queue, bias_vector)
    
    if activation=='relu':
        prg.convolute_2d_relu(queue, (output_matrix_width_height, 
                             output_matrix_width_height,
                             num_filters,), None, c.data, a.data, b.data, d.data, np.int32(filter_dim),
                    np.int32(output_matrix_width_height), np.int32(num_filters), np.int32(num_input_channels),
                    np.int32(padded_matrix_dim))
    elif activation=='sigmoid':
        prg.convolute_2d_sigmoidal(queue, (output_matrix_width_height, 
                             output_matrix_width_height,
                             num_filters,), None, c.data, a.data, b.data, d.data, np.int32(filter_dim),
                    np.int32(output_matrix_width_height), np.int32(num_filters), np.int32(num_input_channels),
                    np.int32(padded_matrix_dim))
    else:
        prg.convolute_2d(queue, (output_matrix_width_height, 
                             output_matrix_width_height,
                             num_filters,), None, c.data, a.data, b.data, d.data, np.int32(filter_dim),
                    np.int32(output_matrix_width_height), np.int32(num_filters), np.int32(num_input_channels),
                    np.int32(padded_matrix_dim))
    
    return c.get()

def maxpooling_2d(input_matrix, kernel_size, stride, prg, queue):
    input_matrix_width_height = len(input_matrix[0])
    input_matrix_depth = len(input_matrix[0][0][0])
    output_matrix_width_height = int(((input_matrix_width_height - kernel_size) / stride) + 1)

    c = cl.array.zeros(queue, shape=(1, output_matrix_width_height,
                                     output_matrix_width_height, input_matrix_depth),
                      dtype=np.float32)
    a = cl.array.to_device(queue, input_matrix)
    
    prg.max_pool_2d(queue, (output_matrix_width_height, 
                         output_matrix_width_height,
                         input_matrix_depth,), None, c.data, a.data, np.int32(kernel_size),
                np.int32(stride), np.int32(output_matrix_width_height), np.int32(input_matrix_depth),
                np.int32(input_matrix_width_height))
    
    return c.get()

def upsampling_2d(input_matrix, kernel_size):
    """
    Increase width and height of input matrix.
    """
    output_matrix = np.repeat(np.repeat(input_matrix, kernel_size, axis=1), kernel_size, axis=2)
    return output_matrix

def reshape(inputs, target_shape):
    print(inputs.shape)
    return np.reshape(inputs, (1,) + target_shape)

def zero_pad_matrix(input_matrix, num_zeros):
    """
    Pad the 3d (nxmxz) input matrix with p zeros
    Assumes 'reshaped' matrix. Need to look at this more closely.
    """
    num_zeros = int(num_zeros)
    return np.pad(input_matrix, ((0, 0), (num_zeros, num_zeros),
                                 (num_zeros, num_zeros), (0, 0)), 'constant')

def build_kernels(ctx):
    return cl.Program(ctx, conv2d_kernel + conv2d_relu_kernel + conv2d_sigmoid_kernel + max_pool_kernel).build()

conv2d_relu_kernel = """__kernel void convolute_2d_relu(__global float *c,
__global const float *a, __global float *b, __constant float *d, int filter_size,
int output_width_height, int num_filters, int num_input_channels, int padded_matrix_dim)
{
    size_t j = get_global_id(0); //height
    size_t i = get_global_id(1); //width
    size_t k = get_global_id(2); //num filters (output depth)
    size_t h;
    size_t l;
    size_t s;
    float conv_result = 0;
    for(h = 0; h < num_input_channels; h++) {
        for(l = 0; l < filter_size; l++) {
            if(num_input_channels == 1) {
                for(s = 0; s < filter_size; s++) {
                    conv_result += b[k + s*num_filters + l*num_filters*filter_size]*a[j+s + (i+l)*padded_matrix_dim];
                }
            } else {
                for(s = 0; s < filter_size; s++) {
                    conv_result += b[k + h*num_filters + s*num_filters*num_input_channels + l*num_filters*num_input_channels*filter_size]*a[h + (j+s)*num_input_channels + (i+l)*num_input_channels*padded_matrix_dim];
                }
            }
        }
    }
    conv_result += d[k];
    if (conv_result < 0) {
        conv_result = 0;
    }
    c[k + j*num_filters + i*num_filters*output_width_height] = conv_result;
}
"""

conv2d_sigmoid_kernel = """__kernel void convolute_2d_sigmoidal(__global float *c,
__global const float *a, __global float *b, __constant float *d, int filter_size,
int output_width_height, int num_filters, int num_input_channels, int padded_matrix_dim)
{
    size_t j = get_global_id(0); //height
    size_t i = get_global_id(1); //width
    size_t k = get_global_id(2); //num filters (output depth)
    size_t h;
    size_t l;
    size_t s;
    float conv_result = 0;
    for(h = 0; h < num_input_channels; h++) {
        for(l = 0; l < filter_size; l++) {
            if(num_input_channels == 1) {
                for(s = 0; s < filter_size; s++) {
                    conv_result += b[k + s*num_filters + l*num_filters*filter_size]*a[j+s + (i+l)*padded_matrix_dim];
                }
            } else {
                for(s = 0; s < filter_size; s++) {
                    conv_result += b[k + h*num_filters + s*num_filters*num_input_channels + l*num_filters*num_input_channels*filter_size]*a[h + (j+s)*num_input_channels + (i+l)*num_input_channels*padded_matrix_dim];
                }
            }
        }
    }
    conv_result += d[k];
    //S(x) = 1 / 1 + e^-x
    conv_result = 1 / (1 + exp(-conv_result));
    c[k + j*num_filters + i*num_filters*output_width_height] = conv_result;
}
"""

conv2d_kernel = """__kernel void convolute_2d(__global float *c,
__global const float *a, __global float *b, __constant float *d, int filter_size,
int output_width_height, int num_filters, int num_input_channels, int padded_matrix_dim)
{
    size_t j = get_global_id(0); //height
    size_t i = get_global_id(1); //width
    size_t k = get_global_id(2); //num filters (output depth)
    size_t h;
    size_t l;
    size_t s;
    float conv_result = 0;
    for(h = 0; h < num_input_channels; h++) {
        for(l = 0; l < filter_size; l++) {
            if(num_input_channels == 1) {
                for(s = 0; s < filter_size; s++) {
                    conv_result += b[k + s*num_filters + l*num_filters*filter_size]*a[j+s + (i+l)*padded_matrix_dim];
                }
            } else {
                for(s = 0; s < filter_size; s++) {
                    conv_result += b[k + h*num_filters + s*num_filters*num_input_channels + l*num_filters*num_input_channels*filter_size]*a[h + (j+s)*num_input_channels + (i+l)*num_input_channels*padded_matrix_dim];
                }
            }
        }
    }
    conv_result += d[k];
    c[k + j*num_filters + i*num_filters*output_width_height] = conv_result;
}
"""

max_pool_kernel = """__kernel void max_pool_2d(__global float *c,
__global const float *a, int kernel_size, int stride,
int output_width_height, int num_filters, int input_w_h)
{
    size_t j = get_global_id(0); //height
    size_t i = get_global_id(1); //width
    size_t k = get_global_id(2); //num filters (output depth)
    int l;
    int s;
    int i_i = i*2;
    int j_j = j*2;
    float curr_max = 0;
    for (l = 0; l < kernel_size; l++) {
        for (s = 0; s < kernel_size; s++) {
            if (curr_max < a[k + (j_j+s)*num_filters + (i_i+l)*num_filters*input_w_h]) {
                curr_max = a[k + (j_j+s)*num_filters + (i_i+l)*num_filters*input_w_h];
            }
        }
    } 

    c[k + j*num_filters + i*num_filters*output_width_height] = curr_max;
}
"""