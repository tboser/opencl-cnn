from __future__ import print_function, division
from naive_prediction.data_utils import get_layer_type
#from tests import predict_with_keras, keras_get_layer_output
from keras import backend as K
#Parallelize prediction of multiple events.
#from joblib import Parallel, delayed

import numpy as np
import pyopencl as cl
import time

def predict_with_keras(model, layer_number, tinput):
    """
    To test other functions
    """
    get_layer_output = K.function([model.layers[layer_number-1].input],
                                   [model.layers[layer_number].output])
    return get_layer_output([tinput][0])

class Predictor:

    def __init__(self, model, cl_idx=0):
        """
        Initialize and store everything necessary to make a prediction.
        """
        self.model = model
        self.layers = []
        self.init_cl(cl_idx)
        self.init_layers()

    def _predict(self, event):
        prediction = cl.array.to_device(self.queue, np.reshape(event, event.shape + (1,)), allocator=self.mem_pool)
        for layer in self.layers:
            prediction = layer.predict(prediction)
        return prediction

    def predict(self, examples, num_jobs=1):
        """
        Make a prediction on examples using the stored model
        """
        #return Parallel(n_jobs=int(num_jobs)) (delayed(_predict) (event, self.layers) for event in examples)
        return [self._predict(event) for event in examples]

    def update_model(self, model):
        pass

    def init_layers(self):
        """
        Initialize layers construct to increase prediction efficiency
        """
        lnum = 0
        for layer in self.model.layers:
            layer_type = get_layer_type(layer)
            layer_config = layer.get_config()
            print(layer_type)
            if layer_type == 'convolution2d':
                self.layers.append(Conv2d(layer, self.prg, self.queue, self.mem_pool))
            if layer_type == 'maxpooling2d':
                self.layers.append(Pool2d(layer, self.prg, self.queue, self.mem_pool))
            if layer_type == 'reshape':
                self.layers.append(Reshape(layer, self.prg, self.queue, self.mem_pool))
            if layer_type == 'upsampling2d':
                self.layers.append(Upsample2d(layer, self.prg, self.queue, self.mem_pool))
            if layer_type == 'dense':
                self.layers.append(Dense(layer, self.prg, self.queue, self.mem_pool, lnum, self.model))
            if layer_type == 'flatten':
                self.layers.append(Flatten(layer, self.prg, self.queue, self.mem_pool, lnum, self.model))
            if layer_type == 'dropout':
                self.layers.append(Dropout(layer, self.prg, self.queue, self.mem_pool, lnum, self.model))
            lnum += 1

    def init_cl(self, cl_idx):
        """
        Initialize pyopencl stuff**
        """
        self.platforms = cl.get_platforms()
        self.devices = self.platforms[0].get_devices()
        self.ctx = cl.Context([self.devices[cl_idx]])
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, conv2d_kernel + conv2d_relu_kernel + conv2d_sigmoid_kernel + max_pool_kernel).build()
        self.mem_pool = cl.tools.MemoryPool(cl.tools.ImmediateAllocator(self.queue))


#### Layer classes
class Conv2d:

    def __init__(self, keras_layer, prg, queue, mem_pool):
        self.layer = keras_layer
        self.prg = prg
        self.queue = queue
        self.stride = 1
        self.activation = self.layer.get_config()['activation']
        self.padding = 'same'
        self.mem_pool = mem_pool
        self.initialize_weights()
        self.pick_kernel()

        self.filter_dim = len(self.weights_matrix)
        self.num_filters = len(self.weights_matrix.T)

    def predict(self, input_matrix):
        start = time.time()
        input_matrix = input_matrix.get()
        if self.padding == 'same':
            num_zeros_padding = (self.filter_dim - 1) / 2
            padded_matrix = self.zero_pad_matrix(input_matrix, num_zeros_padding)
        else:
            padded_matrix = input_matrix
            
        padded_matrix_dim = len(padded_matrix[0][0])

        try:
            num_input_channels = len(input_matrix[0][0][0])
        except:
            num_input_channels = 1
        output_matrix_width_height = padded_matrix_dim - self.filter_dim + 1
        end = time.time()
        print("Conv time for preprocess", end-start)
        
        start = time.time()
        a = cl.array.to_device(self.queue, padded_matrix, allocator=self.mem_pool)
        c = cl.array.zeros(self.queue, shape=(1, output_matrix_width_height,
                                         output_matrix_width_height, self.num_filters),
                          dtype=np.float32, allocator=self.mem_pool)
        end = time.time()
        print("Conv time of cl array stuff", end-start)
        
        start = time.time()
        if self.activation=='relu':
            self.prg.convolute_2d_relu(self.queue, (output_matrix_width_height, 
                                 output_matrix_width_height,
                                 self.num_filters,), None, c.data, a.data, self.weights_matrix.data, self.bias_vector.data, np.int32(self.filter_dim),
                        np.int32(output_matrix_width_height), np.int32(self.num_filters), np.int32(num_input_channels),
                        np.int32(padded_matrix_dim))
        elif self.activation=='sigmoid':
            self.prg.convolute_2d_sigmoidal(self.queue, (output_matrix_width_height, 
                                 output_matrix_width_height,
                                 self.num_filters,), None, c.data, a.data, self.weights_matrix.data, self.bias_vector.data, np.int32(self.filter_dim),
                        np.int32(output_matrix_width_height), np.int32(self.num_filters), np.int32(num_input_channels),
                        np.int32(padded_matrix_dim))
        else:
            self.prg.convolute_2d(self.queue, (output_matrix_width_height, 
                                 output_matrix_width_height,
                                 self.num_filters,), None, c.data, a.data, self.weights_matrix.data, self.bias_vector.data, np.int32(self.filter_dim),
                        np.int32(output_matrix_width_height), np.int32(self.num_filters), np.int32(num_input_channels),
                        np.int32(padded_matrix_dim))
        end = time.time()
        print("Conv time of conv operation", end-start)
        
        return c

    def initialize_weights(self):
        keras_weights = self.layer.get_weights()
        weight_matrix = keras_weights[0]
        bias_vector = keras_weights[1]
        self.weights_matrix = cl.array.to_device(self.queue, weight_matrix, allocator=self.mem_pool)
        self.bias_vector = cl.array.to_device(self.queue, bias_vector, allocator=self.mem_pool)

    def pick_kernel(self):
        if self.activation == 'relu':
            self.convolution_2d = self.prg.convolute_2d_relu
        elif self.activation == 'sigmoid':
            self.convolution_2d = self.prg.convolute_2d_sigmoidal
        else:
            self.convolution_2d = self.prg.convolute_2d

    def zero_pad_matrix(self, input_matrix, num_zeros):
        """
        Pad the 3d (nxmxz) input matrix with p zeros
        Assumes 'reshaped' matrix. Need to look at this more closely.
        """
        num_zeros = int(num_zeros)
        return np.pad(input_matrix, ((0, 0), (num_zeros, num_zeros),
                                     (num_zeros, num_zeros), (0, 0)), 'constant')


class Pool2d:

    def __init__(self, keras_layer, prg, queue, mem_pool):
        self.layer = keras_layer
        self.prg = prg
        self.queue = queue
        self.mem_pool = mem_pool
        self.stride = self.layer.get_config()['strides'][0]
        self.kernel_size = self.layer.get_config()['pool_size'][0]
        self.maxpooling_2d = prg.max_pool_2d

    def predict(self, input_matrix):
        start = time.time()
        input_matrix_width_height = len(input_matrix[0])
        input_matrix_depth = len(input_matrix[0][0][0])
        output_matrix_width_height = int(((input_matrix_width_height - self.kernel_size) / self.stride) + 1)
        end = time.time()
        print("Pooling time to preprocess", end-start)

        start = time.time()
        c = cl.array.zeros(self.queue, shape=(1, output_matrix_width_height,
                                         output_matrix_width_height, input_matrix_depth),
                          dtype=np.float32, allocator=self.mem_pool)
        end = time.time()
        print("Pooling time to create zeros array", end-start)
        
        start = time.time()
        self.prg.max_pool_2d(self.queue, (output_matrix_width_height, 
                             output_matrix_width_height,
                             input_matrix_depth,), None, c.data, input_matrix.data, np.int32(self.kernel_size),
                    np.int32(self.stride), np.int32(output_matrix_width_height), np.int32(input_matrix_depth),
                    np.int32(input_matrix_width_height))
        end = time.time()
        print("Pooling time for actual operation", end-start)
        
        return c

class Reshape:

    def __init__(self, keras_layer, prg, queue, mem_pool):
        self.layer = keras_layer
        self.target_shape = self.layer.get_config()['target_shape']
        self.prg = prg
        self.queue = queue

        self.mem_pool = mem_pool

    def predict(self, input_matrix):
        """
        Reshape matrix
        """
        return input_matrix.reshape((1,) + self.target_shape)

class Upsample2d:

    def __init__(self, keras_layer, prg, queue, mem_pool):
        """
        """
        self.layer = keras_layer
        self.kernel_size = self.layer.get_config()['size'][0]
        self.prg = prg
        self.queue = queue

        self.mem_pool = mem_pool

    def predict(self, input_matrix):
        output_matrix = np.repeat(np.repeat(input_matrix.get(), self.kernel_size, axis=1), self.kernel_size, axis=2)
        return cl.array.to_device(self.queue, output_matrix, allocator=self.mem_pool)

class Dropout:
    """
    Implementation of a dropout layer
    """

    def __init__(self, keras_layer, prg, queue, mem_pool, lnum, model):
        """
        """
        self.layer = keras_layer
        self.prg = prg
        self.queue = queue
        self.mem_pool = mem_pool

        self.lnum = lnum
        self.model = model

    def predict(self, input_matrix):
        """ tmp """
        return cl.array.to_device(self.queue, predict_with_keras(self.model, self.lnum, input_matrix.get()), allocator=self.mem_pool)

class Dense:
    """
    Fully connected layer
    """

    def __init__(self, keras_layer, prg, queue, mem_pool, lnum, model):
        """
        """
        self.layer = keras_layer
        self.prg = prg
        self.queue = queue
        self.mem_pool = mem_pool

        self.lnum = lnum
        self.model = model

    def predict(self, input_matrix):
        """ tmp """
        return cl.array.to_device(self.queue, predict_with_keras(self.model, self.lnum, input_matrix.get()), allocator=self.mem_pool)

class Flatten:
    """
    """

    def __init__(self, keras_layer, prg, queue, mem_pool, lnum, model):
        """
        """
        self.layer = keras_layer
        self.prg = prg
        self.queue = queue
        self.mem_pool = mem_pool

        self.lnum = lnum
        self.model = model

    def predict(self, input_matrix):
        """ tmp """
        return cl.array.to_device(self.queue, predict_with_keras(self.model, self.lnum, input_matrix.get()), allocator=self.mem_pool)












########################################
#### Standalone methods for testing ####
########################################
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
    #print(input_matrix.shape)
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
    #print(inputs.shape)
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
