import mxnet as mx
from mxnet import nd
import numpy as np
from copy import deepcopy
from numpy import random
from scipy.stats import norm


def no_byz(epoch, v, f, lr, perturbation):
    return v


# gaussian attack
def gaussian(epoch, v, f, lr, perturbation):
    if f == 0:
        return v
    else:
        for i in range(f):
            v[i] = mx.nd.random.normal(0, 200, shape=v[i].shape)
    return v


# Backdoor attack
def scale(epoch, v, f, lr, perturbation):
    if f == 0:
        return v
    else:
        scaling_factor = len(v)
        for i in range(f):
            v[i] *= scaling_factor
    return v
