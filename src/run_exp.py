from subprocess import Popen
import numpy as np
import itertools

n_gpus = 8
dropout = [0.1, 0.2, 0.3, 0.4, 0.5]
dropouts = itertools.product(dropout, dropout)
dims = [2**i for i in range(5, 11)]

for i, dims in enumerate(itertools.product(dims[2:4], dims[1:3], dims[4:])):
    if i!=(128,64,512):
        command = ("python src/main.py train --layer-norm --model-path-base grid_search "
                # "--dropouts {} {} "
                # "--tag-dim {} "
                # "--char-dim {} "
                # "--word-dim {} "
                # "--label-dim {} "
                "--h-word {} "
                "--h-char {} "
                "--h-label {} "
                # "--attention-dim {} "
                # "--projection-dim {} "
                "--gpu-id {} "
                 ).format(*dims, (i % n_gpus))
        Popen(command.split())
