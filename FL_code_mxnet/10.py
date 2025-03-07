import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
from copy import deepcopy
import random


def main():

    x = nd.array([[4.7, 2, 3, 4, 5], [3, 2, 7, 1, 0], [5.1, 0.8, 6, 4, 0.7], [5.9, 5, 3, 7, 4]])
    y = nd.array([[5.2, 0, 12, 0, 4], [5, 0.4, 4, 4, 2]])

    arr_org = nd.concat(x, y, dim=0)
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in arr_org]

    print('param_list')
    print(param_list)
    print('========================================')




if __name__ == "__main__":
    main()






