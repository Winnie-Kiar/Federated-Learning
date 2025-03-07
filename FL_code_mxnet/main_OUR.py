import time, logging, os, sys, math, random
import numpy as np
import mxnet as mx
from mxnet import nd, gluon
from mxnet import autograd as ag
import nd_aggregation
import byzantine
from copy import deepcopy
from mxnet.gluon.model_zoo import vision as models

import argparse

np.warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="dataset", type=str, default='mnist')
parser.add_argument("--classes", type=int, help="number of classes", default=10)
parser.add_argument("--batchsize", type=int, help="batchsize", default=32)
parser.add_argument("--lr", type=float, help="learning rate", default=0.001)
parser.add_argument("--bias", help="way to assign data to workers", type=float, default=0.5)
parser.add_argument("--gpu", help="index of gpu", type=int, default=0)
parser.add_argument("--model", type=str, help="model", default='cnn')
parser.add_argument("--momentum", type=float, help="momentum", default=0)

parser.add_argument("--seed", type=int, help="random seed", default=733)
parser.add_argument("--interval", type=int, help="log interval (epochs)", default=10)
parser.add_argument("--log", type=str, help="dir of the log file", default='log.txt')

parser.add_argument("--epochs", type=int, help="number of epochs", default=500)
parser.add_argument("--local_round", help="number of local rounds", type=int, default=1)

parser.add_argument("--nworkers", type=int, help="number of workers", default=10)
parser.add_argument("--nbyz", type=int, help="number of Byzantine workers", default=2)
parser.add_argument("--byz_type", type=str, help="type of Byzantine workers, 'none', 'labelflip', , 'gaussian'", default='none')
parser.add_argument("--aggregation", type=str, help="'mean', 'Guard' ", default='mean')

parser.add_argument("--perturbation", type=str, help="'sgn', 'uv', 'std'", default='sgn')


args = parser.parse_args()

input_str = ' '.join(sys.argv)
print(input_str)

filehandler = logging.FileHandler(args.log)
streamhandler = logging.StreamHandler()

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

if args.gpu == -1:
    context = mx.cpu()
else:
    context = mx.gpu(args.gpu)


if args.dataset == 'mnist':
    num_inputs = 28 * 28


with context:

    # set random seed
    mx.random.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    batch_size = args.batchsize
    classes = args.classes
    num_workers = args.nworkers
    local_round = args.local_round
    nbyz = args.nbyz

    perturbation = args.perturbation

    test_acc_list = []



    def evaluate_accuracy(data_iterator, net, trigger=False, target=None):
        acc = mx.metric.Accuracy()
        for i, (data, label) in enumerate(data_iterator):
            data = data.as_in_context(context)
            label = label.as_in_context(context)
            remaining_idx = list(range(data.shape[0]))

            if trigger and (args.dataset == 'mnist' or args.dataset == 'Fashion'):
                for example_id in range(data.shape[0]):
                    data[example_id][0][26][26] = 1
                    data[example_id][0][24][26] = 1
                    data[example_id][0][26][24] = 1
                    data[example_id][0][25][25] = 1
                for example_id in range(data.shape[0]):
                    if label[example_id] != target:
                        label[example_id] = target
                    else:
                        remaining_idx.remove(example_id)

            output = net(data)
            predictions = nd.argmax(output, axis=1)

            predictions = predictions[remaining_idx]
            label = label[remaining_idx]

            acc.update(preds=predictions, labels=label)

        return acc.get()[1]



    def load_model(model_name):
        if model_name == 'cnn':
            net = gluon.nn.Sequential()
            with net.name_scope():
                net.add(gluon.nn.Conv2D(channels=30, kernel_size=3, activation='relu'))
                net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
                net.add(gluon.nn.Conv2D(channels=50, kernel_size=3, activation='relu'))
                net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
                net.add(gluon.nn.Flatten())
                net.add(gluon.nn.Dense(100, activation="relu"))
                net.add(gluon.nn.Dense(classes))
        elif model_name == 'mlr':
            net = gluon.nn.Sequential()
            with net.name_scope():
                net.add(gluon.nn.Dense(classes))
        elif model_name == 'resnet':
            kwargs = {'classes': 10, 'thumbnail': True}
            res_layers = [3, 3, 3]
            res_channels = [16, 16, 32, 64]
            resnet_class = models.ResNetV1
            block_class = models.BasicBlockV1
            net = resnet_class(block_class, res_layers, res_channels, **kwargs)
            net.initialize(mx.init.Xavier(magnitude=3.1415926), ctx=context)

        if args.model != 'resnet':
            net.initialize(mx.init.Xavier(), ctx=context)
        return net

    model_name = args.model
    net = load_model(model_name)


    #----------------------------------
    paraString = str(args.seed) +"+" +str(args.dataset) +"+" + str(args.model) + "+" + "bias" + str(args.bias) + "+epoch" + str(args.epochs)+\
                 "+" + "local" + str(args.local_round) + "+lr" + str(args.lr) + "+" + "batch" + str(args.batchsize) + \
                 "+"+"nwork" + str(args.nworkers) + "+" + "nbyz" + str(args.nbyz) + "+" + str(args.byz_type) + \
                 "+" + str(args.aggregation) + "+" + str(perturbation) + ".txt"
    #----------------------------------

    optimizer = 'sgd'
    lr = args.lr
    optimizer_params = {'momentum': args.momentum, 'learning_rate': lr, 'wd': 0.0001}


    loss_func = gluon.loss.SoftmaxCrossEntropyLoss()

    train_metric = mx.metric.Accuracy()

    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    train_cross_entropy = mx.metric.CrossEntropy()



    # ------------------------- DATA PART ------------------------------------
    if (args.dataset == 'mnist' and args.model == 'cnn'):
        def transform(data, label):
            return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)
        train_data_all = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform), 60000, shuffle=True, last_batch='rollover')
        test_test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform), 256, shuffle=False, last_batch='rollover')

    elif (args.dataset == 'Fashion' and args.model == 'cnn'):
        def transform(data, label):
            return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)
        train_data_all = mx.gluon.data.DataLoader(mx.gluon.data.vision.FashionMNIST(train=True, transform=transform), 60000, shuffle=True, last_batch='rollover')
        test_test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.FashionMNIST(train=False, transform=transform), 256, shuffle=False, last_batch='rollover')


    elif (args.dataset == 'mnist' and args.model == 'mlr'):
        def transform(data, label):
            return data.astype(np.float32) / 255, label.astype(np.float32)
        train_data_all = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform), 60000, shuffle=True, last_batch='rollover')
        test_test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform), 256, shuffle=False, last_batch='rollover')


    elif (args.dataset == 'cifar10'):
        def transform(data, label):
            return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255., label.astype(np.float32)
        train_data_all = mx.gluon.data.DataLoader(mx.gluon.data.vision.CIFAR10(train=True, transform=transform), 50000, shuffle=True, last_batch='rollover')
        test_test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.CIFAR10(train=False, transform=transform), 256, shuffle=False, last_batch='rollover')



    # biased assignment
    bias_weight = args.bias
    other_group_size = (1 - bias_weight) / 9.
    worker_per_group = num_workers / 10

    # assign training data to each worker
    each_worker_data = [[] for _ in range(num_workers)]
    each_worker_label = [[] for _ in range(num_workers)]


    def transform_training(train_x):
        transformed_x = train_x.copy()
        n_x = train_x.shape[0]
        start_x = np.random.randint(9, size=(n_x,))
        start_y = np.random.randint(9, size=(n_x,))
        to_flip = nd.random.uniform(shape=(n_x,))
        padded = nd.pad(transformed_x, mode="constant", constant_value=0, pad_width=(0, 0, 0, 0, 4, 4, 4, 4))
        for i in range(n_x):
            cropped = padded[i][:, start_x[i]:start_x[i] + 32, start_y[i]:start_y[i] + 32]
            if to_flip[i] > 0.5:
                transformed_x[i] = cropped[:, :, ::-1].copy()
            else:
                transformed_x[i] = cropped.copy()
        return transformed_x


    for _, (data, label) in enumerate(train_data_all):
        for (x, y) in zip(data, label):

            if args.dataset == 'mnist' and args.model == 'mlr':
                x = x.as_in_context(context).reshape(-1, num_inputs)
            elif args.dataset == 'mnist' and args.model == 'cnn':
                x = x.as_in_context(context).reshape(1, 1, 28, 28)
            elif args.dataset == 'Fashion'and args.model == 'cnn':
                x = x.as_in_context(context).reshape(1, 1, 28, 28)
            elif args.dataset == 'cifar10':
                x = x.as_in_context(context).reshape(1, 3, 32, 32)

            y = y.as_in_context(context)

            upper_bound = (y.asnumpy()) * (1 - bias_weight) / 9. + bias_weight
            lower_bound = (y.asnumpy()) * (1 - bias_weight) / 9.
            rd = np.random.random_sample()

            if rd > upper_bound:
                worker_group = int(np.floor((rd - upper_bound) / other_group_size) + y.asnumpy() + 1)
            elif rd < lower_bound:
                worker_group = int(np.floor(rd / other_group_size))
            else:
                worker_group = y.asnumpy()

            rd = np.random.random_sample()
            selected_worker = int(worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
            each_worker_data[selected_worker].append(x)
            each_worker_label[selected_worker].append(y)


    each_worker_data = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_data]
    each_worker_label = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_label]

    random_order = np.random.RandomState(seed=args.seed).permutation(num_workers)
    each_worker_data = [each_worker_data[i] for i in random_order]
    each_worker_label = [each_worker_label[i] for i in random_order]


    train_data = gluon.data.DataLoader(mx.gluon.data.dataset.ArrayDataset(each_worker_data[0], each_worker_label[0]),
                                      batch_size=args.batchsize, shuffle=True, last_batch='rollover', num_workers=0)

    test_train_data = gluon.data.DataLoader(mx.gluon.data.dataset.ArrayDataset(each_worker_data[0], each_worker_label[0]),
                                      batch_size=args.batchsize, shuffle=False, last_batch='rollover', num_workers=0)

    # ------------------------- DATA PART ------------------------------------


    # warmup
    print('warm up')

    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)

    trainer.set_learning_rate(0.001)

    for local_epoch in range(1):
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(context)
            label = label.as_in_context(context)

            with ag.record():
                outputs = net(data)
                loss = loss_func(outputs, label)
            loss.backward()
            trainer.step(args.batchsize)
            break

    nd.waitall()


    #################################### Adding triggers ###########################
    '''
    Here: add triggers
    '''
    # add triggered training examples. we follow badnets and use p=1. the target label is set to be 0
    if args.byz_type == 'scale':
        if args.dataset == 'mnist' or args.dataset == 'Fashion':
            for worker_id in range(args.nbyz):
                # duplicate training data for each compromised worker
                each_worker_data[worker_id] = nd.repeat(each_worker_data[worker_id], repeats=2, axis=0)
                # add the trigger to one of the two duplicates, specifically, we set the feature value to 0 every 20 features
                for example_id in range(0, each_worker_data[worker_id].shape[0], 2):
                    # the trigger is the same as that used in badnets
                    each_worker_data[worker_id][example_id][0][26][26] = 1
                    each_worker_data[worker_id][example_id][0][24][26] = 1
                    each_worker_data[worker_id][example_id][0][26][24] = 1
                    each_worker_data[worker_id][example_id][0][25][25] = 1
                # duplicate training label for each compromised worker
                each_worker_label[worker_id] = nd.repeat(each_worker_label[worker_id], repeats=2, axis=0)
                # set the target label
                for example_id in range(0, each_worker_label[worker_id].shape[0], 2):
                    each_worker_label[worker_id][example_id] = 0
            backdoor_target = 0

    ######################################################################


    tic = time.time()
    # reset optimizer
    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)

    if args.byz_type == 'label':
        for i in range(args.nbyz):
            for j in range(len(each_worker_label[i])):
                if args.dataset == 'mnist' or args.dataset == 'Fashion' or args.dataset == 'cifar10':
                    each_worker_label[i][j] = 9 - each_worker_label[i][j]


    for each_epoch in range(args.epochs):

        # Do not attack in the first epoch
        if each_epoch > 0:
            # byzantine
            if args.byz_type == 'gauss':
                byz = byzantine.gaussian

            elif args.byz_type == 'scale':
                byz = byzantine.scale

            elif args.byz_type == 'label' or args.byz_type == 'none':
                byz = byzantine.no_byz
            else:
                print('Not found')


        else:
            byz = byzantine.no_byz

        trainer.set_learning_rate(lr)

        selected_worker_index = [i for i in range(num_workers)]  # default select all workers

        grad_list = []

        for each_worker in selected_worker_index:

            minibatch = np.random.choice(list(range(each_worker_data[each_worker].shape[0])), size=args.batchsize, replace=False)
            if args.dataset == 'mnist' or args.dataset == 'Fashion':
                data = each_worker_data[each_worker][minibatch].as_in_context(context)
                label = each_worker_label[each_worker][minibatch].as_in_context(context)

            # compute gradient
            with ag.record():
                outputs = net(data)
                loss = loss_func(outputs, label)
            loss.backward()

            grad_list.append([param.grad().copy() for param in net.collect_params().values() if param.grad_req != 'null'])

        # aggregation
        if args.aggregation == 'mean':
            agg_gradient = nd_aggregation.mean(each_epoch, grad_list, net, lr / args.batchsize, perturbation, nbyz, byz)
        elif args.aggregation == 'trim':
            agg_gradient = nd_aggregation.trim(each_epoch, grad_list, net, lr / args.batchsize, perturbation, nbyz, byz)
        elif args.aggregation == 'median':
            agg_gradient = nd_aggregation.median(each_epoch, grad_list, net, lr / args.batchsize, perturbation, nbyz, byz)
        elif args.aggregation == 'krum':
            agg_gradient = nd_aggregation.krum(each_epoch, grad_list, net, lr / args.batchsize, perturbation, nbyz, byz)


        del grad_list

        # update the global model
        if args.dataset == 'mnist' or args.dataset == 'Fashion':
            idx = 0
            for j, (param) in enumerate(net.collect_params().values()):
                if param.grad_req == 'null':
                    continue
                param.set_data(param.data() - lr / args.batchsize * agg_gradient[idx:(idx + param.data().size)].reshape(param.data().shape))
                idx += param.data().size


        # validation
        if each_epoch % args.interval == 0 or each_epoch == args.epochs - 1:
            if args.byz_type == 'scale':
                test_accuracy = evaluate_accuracy(test_test_data, net)
                backdoor_acc = evaluate_accuracy(test_test_data, net, trigger=True, target=backdoor_target)
                test_acc_list.append((test_accuracy, backdoor_acc))
                print('[Epoch %d] acc-top1=%.4f, backdoor_acc=%.4f' % (each_epoch, test_accuracy, backdoor_acc))

            else:
                test_accuracy = evaluate_accuracy(test_test_data, net)
                test_acc_list.append(test_accuracy)
                print('[Epoch %d] acc-top1=%.4f' % (each_epoch, test_accuracy))


            np.savetxt('out/' + paraString, test_acc_list, fmt='%.4f')

            tic = time.time()

            nd.waitall()

        np.savetxt('out/' + paraString, test_acc_list, fmt='%.4f')