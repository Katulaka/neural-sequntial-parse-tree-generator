from subprocess import Popen
import numpy as np
import itertools

proc = []
n_gpus = 8
s_gpu = 1
dropouts = [0.1, 0.2, 0.3, 0.4]
iter = itertools.product([0.2], dropouts)
for i, drops in enumerate(iter):
    path = ("models/tf-with-new-dims-n-tags-dropouts[{},{}]").format(*drops)
    device = 'GPU:{}'.format(i % n_gpus + s_gpu)
    command = ("python src/main.py train --layer-norm "
                "--model-path-base {} "
                "--dropouts {} {} "
                "--gpu-id {} "
                 ).format(path, *drops, device)
    proc.append(Popen(command.split()))
