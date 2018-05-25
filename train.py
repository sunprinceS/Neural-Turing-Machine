import argparse
import json
import random
import sys
import torch
import numpy as np
from tasks.copy import TaskCopy

## Marcos
RANDOM_SEED = 1000

def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='train.py',
            description = 'Neural Turing Machine Training')
    parser.add_argument('--seed',type=int,default=RANDOM_SEED)
    parser.add_argument('--task',type=str,choices=['copy'],default='copy')
    parser.add_argument('--seq_width',type=int,help='# of bits per time slot')
    parser.add_argument('--ctrl_size',type=int, help='hidden dimension of controller')
    parser.add_argument('--ctrl_num_layers',type=int,help='# of layers of controller')
    parser.add_argument('--mem_size',type=int,help='# of memory cells')
    parser.add_argument('--mem_dim',type=int,help='dimension of memory cells')
    parser.add_argument('--num_heads',type=int,help='# of heads (W = R)')
    parser.add_argument('--batch',type=int,help='# of batch')
    parser.add_argument('--batch_size',type=int,help = '# of datum per batch')
    parser.add_argument('--mark',type=str,default='sun',help='mark for running (DEBUG)')

    args = parser.parse_args()

    init_seed(args.seed)
    task = TaskCopy(vars(args))
    task.init()
    task.train()
