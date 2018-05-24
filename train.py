import argparse
import json
import random
import sys
import numpy as np
import torch
from ntm import *
from tasks.copy import TaskCopy

RANDOM_SEED = 1000

def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='train.py')
    parser.add_argument('--seed',type=int,default=RANDOM_SEED)
    parser.add_argument('--task',type=str,choices=['copy'],default='copy')
    parser.add_argument('--seq_width',type=int)
    parser.add_argument('--ctrl_size',type=int)
    parser.add_argument('--ctrl_num_layers',type=int)
    parser.add_argument('--mem_size',type=int)
    parser.add_argument('--mem_dim',type=int)
    parser.add_argument('--num_heads',type=int)
    parser.add_argument('--batch',type=int)
    parser.add_argument('--batch_size',type=int)
    parser.add_argument('--epoch',type=int)



    args = parser.parse_args()

    init_seed(args.seed)
    task = TaskCopy(vars(args))
    task.train()
