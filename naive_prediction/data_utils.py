"""
Some helper methods to make the transition from keras model to opencl predictions smoother
"""

import json
from keras import models, layers

def write_keras_model_to_file(keras_model, file_path):
    """
    Save a keras model to path file_path
    """
    keras_model.save(file_path)

def read_keras_model_from_file(file_path):
    """
    Return a keras model saved at path file_path
    """
    return models.load_model(file_path)

def get_layer_type(keras_layer):
    """
    return string describing the type of a layer
    """
    config = keras_layer.get_config()
    return config['name'].split('_', 1)[0]