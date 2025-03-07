import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
from copy import deepcopy
import random


def no_byz(epoch, v, f, lr, perturbation):
    return v

# average aggregation rule
def mean(epoch, gradients, net, lr, perturbation, f=0, byz=no_byz):
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    param_list = byz(epoch, param_list, f, lr, perturbation)
    median_nd = nd.mean(nd.concat(*param_list, dim=1), axis=-1)
    return median_nd

# coordinate-wise trimmed mean aggregation rule
def trim(epoch, gradients, net, lr, perturbation, f=0, byz=no_byz):
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    param_list = byz(epoch, param_list, f, lr, perturbation)
    sorted_array = nd.sort(nd.concat(*param_list, dim=1), axis=-1)
    n = len(param_list)
    b = f
    m = n - b * 2
    median_nd = nd.mean(sorted_array[:, b:(b + m)], axis=-1, keepdims=1)
    return median_nd

# coordinate-wise median aggregation rule
def median(epoch, gradients, net, lr, perturbation, f=0, byz=no_byz):
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    param_list = byz(epoch, param_list, f, lr, perturbation)


    sorted_array = nd.sort(nd.concat(*param_list, dim=1), axis=-1)
    if sorted_array.shape[-1] % 2 == 1:
        median_nd = sorted_array[:, int(sorted_array.shape[-1] / 2)]
    else:
        median_nd = (sorted_array[:, int((sorted_array.shape[-1] / 2 - 1))] + sorted_array[:,int((sorted_array.shape[-1] / 2))]) / 2
    return median_nd


def score(gradient, v, f):
    num_neighbours = v.shape[1] - 2 - f
    sorted_distance = nd.square(v - gradient).sum(axis=0).sort()
    return nd.sum(sorted_distance[1:(1 + num_neighbours)]).asscalar()

# Krum aggregation rule
def krum(epoch, gradients, net, lr, perturbation, f=0, byz=no_byz):
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    num_params = len(param_list)
    q = f
    if num_params - f - 2 <= 0:
        q = num_params - 3
    param_list = byz(epoch, param_list, f, lr, perturbation)

    v = nd.concat(*param_list, dim=1)
    scores = nd.array([score(gradient, v, q) for gradient in param_list])
    min_idx = int(scores.argmin(axis=0).asscalar())
    krum_nd = nd.reshape(param_list[min_idx], shape=(-1,))
    return krum_nd

