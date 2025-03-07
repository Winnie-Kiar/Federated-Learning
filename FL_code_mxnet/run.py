import os

dataset = ['mnist']   # 'mnist', 'Fashion', 'cifar10'
model = ['cnn']      # 'cnn', 'mlr', 'resnet'
bias = [0.5]
seed = [0]
epochs = [2550]
local_round = [10]
lr = [0.01]
batchsize = [32]

nworkers = [100]
nbyz = [20] 

byz_type = ['gauss']  # 'none', 'gauss', 'label', 'scale'
aggregation = ['mean'] # 'mean', 'trim', 'median', 'krum'

perturbation = ['sgn']   # 'sgn', 'uv', 'std' (not use)

gpu = [0]  # if you do not have gpu, set gpu = [-1]

for each_seed in seed:
    for each_dataset in dataset:
        for each_model in model:
            for each_bias in bias:
                for each_local_round in local_round:
                    for each_lr in lr:
                        for each_batchsize in batchsize:
                            for each_nworkers in nworkers:
                                for each_nbyz in nbyz:
                                    for each_byz_type in byz_type:
                                        for each_aggregation in aggregation:
                                            for each_perturbation in perturbation:
                                                suffix = "python main_OUR.py" \
                                                    + " --dataset=" + str(each_dataset)  \
                                                    + " --model=" + str(each_model) \
                                                    + " --bias=" + str(each_bias) \
                                                    + " --seed=" + str(each_seed) \
                                                    + " --epochs=" + str(epochs[0]) \
                                                    + " --local_round=" + str(each_local_round) \
                                                    + " --lr=" + str(each_lr) \
                                                    + " --batchsize=" + str(each_batchsize) \
                                                    + " --nworkers=" + str(each_nworkers) \
                                                    + " --nbyz=" + str(each_nbyz) \
                                                    + " --byz_type=" + str(each_byz_type) \
                                                    + " --aggregation=" + str(each_aggregation) \
                                                    + " --perturbation=" + str(each_perturbation) \
                                                    + " --gpu=" + str(gpu[0])
                                                os.system(suffix)