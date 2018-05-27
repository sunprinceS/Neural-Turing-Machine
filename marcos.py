import numpy as np
import torch
import random
from tasks.copy import TaskCopy
from tasks.repeatcopy import TaskRepeatCopy

## Marcos
RANDOM_SEED = 1000
TASK_DICT={'copy':TaskCopy,'repeat-copy':TaskRepeatCopy}

def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

