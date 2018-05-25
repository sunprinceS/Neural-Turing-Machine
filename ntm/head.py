import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def _split_cols(mat, lengths):
    """
    Args:
        mat: 2D tensor, shape = (N,sum(len of param))
        lengths: list of length of k, β, g, s, γ, (e, a)
    Outputs:
        list of 2D tensor corresponding to the providing param lengths
    """
    assert mat.size()[1] == sum(lengths)
    l = np.cumsum([0] + lengths)
    results = []
    for s, e in zip(l[:-1], l[1:]):
        results += [mat[:, s:e]]
    return results


class NTMHeadBase(nn.Module):

    def __init__(self, memory,ctrl_size):
        super(NTMHeadBase, self).__init__()

        self.memory = memory
        self.N, self.M = memory.size
        self.ctrl_size = ctrl_size

    def _address_memory(self, k, beta, g, s, gamma, w_prev):
        """Preprocess the ctrl output to match the valid range & get the address weighting
            beta: [0,∞)
            g: (0,1)
            s: (0,1), and sum(s,dim=1) = 1.0
            gamma: [1,∞)
            w_prev: address weighting of t-1, shape = (batch_size,N)
        """
        k = k.clone()
        beta = F.relu(beta)
        g = F.sigmoid(g)
        s = F.softmax(s, dim=1)
        gamma = 1 + F.softplus(gamma)

        w = self.memory.address(k, beta, g, s, gamma, w_prev)

        return w


class NTMReadHead(NTMHeadBase):
    def __init__(self, memory,ctrl_size):
        super(NTMReadHead, self).__init__(memory, ctrl_size)

        # Corresponding to k, beta, g, s, gamma sizes from the paper
        self.read_chunk_lens= [self.M, 1, 1, 3, 1]
        self.fc_read = nn.Linear(ctrl_size, sum(self.read_chunk_lens))
        self.reset_parameters()

    def create_new_state(self, batch_size):
        """ Init new state and batchify """

        return torch.zeros(batch_size, self.N)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_read.weight, gain=1.4)
        nn.init.normal_(self.fc_read.bias, std=0.01)

    def is_read_head(self):
        return True

    def forward(self, embeddings, w_prev):
        o = self.fc_read(embeddings)
        k, beta, g, s, gamma = _split_cols(o, self.read_chunk_lens)

        w = self._address_memory(k, beta, g, s, gamma, w_prev)
        r = self.memory.read(w)

        return r, w


class NTMWriteHead(NTMHeadBase):
    def __init__(self, memory, controller_size):
        super(NTMWriteHead, self).__init__(memory, controller_size)

        # Corresponding to k, beta, g, s, gamma, e, a sizes from the paper
        self.write_chunk_lens = [self.M, 1, 1, 3, 1, self.M, self.M]
        self.fc_write = nn.Linear(controller_size, sum(self.write_chunk_lens))
        self.reset_parameters()

    def create_new_state(self, batch_size):
        """ Init new state and batchify """

        return torch.zeros(batch_size, self.N)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_write.weight, gain=1.4)
        nn.init.normal_(self.fc_write.bias, std=0.01)

    def is_read_head(self):
        return False

    def forward(self, embeddings, w_prev):
        o = self.fc_write(embeddings)
        k, beta, g, s, gamma, e, a = _split_cols(o, self.write_chunk_lens)

        # e should be in [0, 1]
        e = F.sigmoid(e)

        # Write to memory
        w = self._address_memory(k, beta, g, s, gamma, w_prev)
        self.memory.write(w, e, a)

        return w
