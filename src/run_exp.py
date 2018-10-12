from subprocess import Popen
import numpy as np
import itertools

proc = []
n_gpus = 8
dropout = [0.1, 0.2, 0.3, 0.4, 0.5]
dropouts = itertools.product(dropout, dropout)
dims = [2**i for i in range(5, 11)]

for i, dims in enumerate(itertools.product(dims[1:3], dims[1:4])):
    if i!=(64,64):
        command = ("python src/main.py train --layer-norm --model-path-base grid_search "
                # "--dropouts {} {} "
                "--tag-dim {} "
                # "--char-dim {} "
                # "--word-dim {} "
                "--label-dim {} "
                # "--h-char {} "
                # "--h-word {} "
                # "--h-label {} "
                # "--attention-dim {} "
                # "--projection-dim {} "
                "--gpu-id {} "
                 ).format(*dims, (i % n_gpus))
    proc.append(Popen(command.split()))
