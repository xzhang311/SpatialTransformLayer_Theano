import sys
sys.path.append('..')

import os
import json
from time import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.externals import joblib

import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv

from lib import activations
from lib import updates
from lib import inits
from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, l2normalize
from lib.metrics import nnc_score, nnd_score
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data, center_crop, patch
import cPickle

def Repeat(x, n_repeats):
    rep = T.ones((n_repeats,), dtype='int32').dimshuffle('x', 0)
    x = T.dot(x.reshape((-1, 1)), rep)
    return x.flatten()


def Interpolate(im, x, y, downsample_factor):
    # constants
    num_batch, height, width, channels = im.shape
    height_f = T.cast(height, 'float32')
    width_f = T.cast(width, 'float32')
    out_height = T.cast(height_f // downsample_factor, 'int64')
    out_width = T.cast(width_f // downsample_factor, 'int64')
    zero = T.zeros([], dtype='int64')
    max_y = T.cast(im.shape[1] - 1, 'int64')
    max_x = T.cast(im.shape[2] - 1, 'int64')

    # scale indices from [-1, 1] to [0, width/height]
    x = (x + 1.0)*(width_f) / 2.0
    y = (y + 1.0)*(height_f) / 2.0

    # do sampling
    x0 = T.cast(T.floor(x), 'int64')
    x1 = x0 + 1
    y0 = T.cast(T.floor(y), 'int64')
    y1 = y0 + 1

    x0 = T.clip(x0, zero, max_x)
    x1 = T.clip(x1, zero, max_x)
    y0 = T.clip(y0, zero, max_y)
    y1 = T.clip(y1, zero, max_y)
    dim2 = width
    dim1 = width*height
    base = Repeat(
        T.arange(num_batch, dtype='int32')*dim1, out_height*out_width)
    base_y0 = base + y0*dim2
    base_y1 = base + y1*dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # use indices to lookup pixels in the flat image and restore channels dim
    im_flat = im.reshape((-1, channels))
    Ia = im_flat[idx_a]
    Ib = im_flat[idx_b]
    Ic = im_flat[idx_c]
    Id = im_flat[idx_d]

    # and finanly calculate interpolated values
    x0_f = T.cast(x0, 'float32')
    x1_f = T.cast(x1, 'float32')
    y0_f = T.cast(y0, 'float32')
    y1_f = T.cast(y1, 'float32')
    wa = ((x1_f-x) * (y1_f-y)).dimshuffle(0, 'x')
    wb = ((x1_f-x) * (y-y0_f)).dimshuffle(0, 'x')
    wc = ((x-x0_f) * (y1_f-y)).dimshuffle(0, 'x')
    wd = ((x-x0_f) * (y-y0_f)).dimshuffle(0, 'x')
    output = T.sum([wa*Ia, wb*Ib, wc*Ic, wd*Id], axis=0)
    return output


def Linspace(start, stop, num):
    start = T.cast(start, 'float32')
    stop = T.cast(stop, 'float32')
    num = T.cast(num, 'float32')
    step = (stop-start)/(num-1)
    return T.arange(num, dtype='float32')*step+start


def Meshgrid(height, width):
    x_t = T.dot(T.ones((height, 1)),
                Linspace(-1.0, 1.0, width).dimshuffle('x', 0))
    y_t = T.dot(Linspace(-1.0, 1.0, height).dimshuffle(0, 'x'),
                T.ones((1, width)))

    x_t_flat = x_t.reshape((1, -1))
    y_t_flat = y_t.reshape((1, -1))
    ones = T.ones_like(x_t_flat)
    grid = T.concatenate([x_t_flat, y_t_flat, ones], axis=0)
    return grid

# full connect layer to computer transformation params
def GetTheta(X, w1, g1, b1, w2, g2, b2):
    relu = activations.Rectify()
    X = X.flatten()
    o1 = relu(batchnorm(T.dot(X, w1), g=g1, b=b1))
    theta = relu(batchnorm(T.dot(o1, w2), g=g2, b=g2)) # In original impl, it nonlinearity and w is constant zero.
    return theta

def Transform(X, w1, g1, b1, w2, g2, b2, downsample_factor=2):
    theta = GetTheta(X, w1, g1, b1, w2, g2, b2)
    num_batch, num_channels, height, width = X.shape
    theta = T.reshape(theta, (-1, 2, 3))

    height_f = T.cast(height, 'float32')
    width_f = T.cast(width, 'float32')
    out_height = T.cast(height_f // downsample_factor, 'int64')
    out_width = T.cast(width_f // downsample_factor, 'int64')
    grid = Meshgrid(out_height, out_width)

    # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
    T_g = T.dot(theta, grid)
    x_s, y_s = T_g[:, 0], T_g[:, 1]
    x_s_flat = x_s.flatten()
    y_s_flat = y_s.flatten()

    # dimshuffle input to  (bs, height, width, channels)
    input_dim = input.dimshuffle(0, 2, 3, 1)
    input_transformed = Interpolate(input_dim, x_s_flat, y_s_flat, downsample_factor)

    output = T.reshape(input_transformed,
                       (num_batch, out_height, out_width, num_channels))

    output = output.dimshuffle(0, 3, 1, 2)
    return output