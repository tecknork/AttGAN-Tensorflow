import functools
from collections import defaultdict

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from torchvision.transforms import transforms
import torch
import tqdm
# ==============================================================================
# =                                 operations                                 =
# ==============================================================================

def tile_concat(a_list, b_list=[]):
    # tile all elements of `b_list` and then concat `a_list + b_list` along the channel axis
    # `a` shape: (N, H, W, C_a)
    # `b` shape: can be (N, 1, 1, C_b) or (N, C_b)
    a_list = list(a_list) if isinstance(a_list, (list, tuple)) else [a_list]
    b_list = list(b_list) if isinstance(b_list, (list, tuple)) else [b_list]
    for i, b in enumerate(b_list):
        b = tf.reshape(b, [-1, 1, 1, b.shape[-1]])
        b = tf.tile(b, [1, a_list[0].shape[1], a_list[0].shape[2], 1])
        b_list[i] = b
    return tf.concat(a_list + b_list, axis=-1)


def repeat_tensor(tensor, axis, multiple):
    """e.g. (1,2,3)x3 = (1,1,1,2,2,2,3,3,3)"""

    result_shape = tensor.shape.as_list()
    for i, v in enumerate(result_shape):
        if v is None:
            result_shape[i] = tf.shape(tensor)[i]
    result_shape[axis] *= multiple

    tensor = tf.expand_dims(tensor, axis + 1)
    mul = [1] * len(tensor.shape)
    mul[axis + 1] = multiple
    tensor = tf.tile(tensor, mul)
    tensor = tf.reshape(tensor, result_shape)

    return tensor


def tile_tensor(tensor, axis, multiple):
    """e.g. (1,2,3)x3 = (1,2,3,1,2,3,1,2,3)"""
    mul = [1] * len(tensor.shape)
    mul[axis] = multiple

    return tf.tile(tensor, mul)


# ==============================================================================
# =                                   others                                   =
# ==============================================================================

def get_norm_layer(norm, training, updates_collections=None):
    if norm == 'none':
        return lambda x: x
    elif norm == 'batch_norm':
        return functools.partial(slim.batch_norm, scale=True, is_training=training, updates_collections=updates_collections)
    elif norm == 'instance_norm':
        return slim.instance_norm
    elif norm == 'layer_norm':
        return slim.layer_norm

def get_dimensions(array, level=0):
    yield level, len(array)
    try:
        for row in array:
            yield from get_dimensions(row, level + 1)
    except TypeError: #not an iterable
        pass

def get_max_shape(array):
    dimensions = defaultdict(int)
    for level, length in get_dimensions(array):
        dimensions[level] = max(dimensions[level], length)
    return [value for _, value in sorted(dimensions.items())]
def iterate_nested_array(array, index=()):
    try:
        for idx, row in enumerate(array):
            yield from iterate_nested_array(row, (*index, idx))
    except TypeError: # final level
        for idx, item in enumerate(array):
            yield (*index, idx), item

def pad(array, fill_value):
            dimensions = get_max_shape(array)
            result = np.full(dimensions, fill_value)
            for index, value in iterate_nested_array(array):
                result[index] = value
            return result


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def imagenet_transform():
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return transform