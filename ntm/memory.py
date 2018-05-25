import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

def _convolve(w,s):
    pad_w = torch.cat([w[-1:],w,w[:1]])
    return F.conv1d(pad_w.view(1,1,-1),s.view(1,1,-1)).view(-1)

class NTMMemory(nn.Module):
    def __init__(self, N, M):
        super(NTMMemory, self).__init__()

        self.N = N
        self.M = M

        # mem_init not training
        self.register_buffer('mem_init', torch.Tensor(N, M))
        self.reset_parameters()

    def reset_parameters(self):
        stdev = 1 / (np.sqrt(self.N + self.M))
        nn.init.uniform_(self.mem_init, -stdev, stdev)

    def create_new_state(self, batch_size):
        """ Init new memory and batchify """

        self.batch_size = batch_size
        self.memory = self.mem_init.clone().repeat(batch_size, 1, 1)

    def get_memory(self):
        return self.memory.clone().squeeze(0)

    @property
    def size(self):
        return self.N, self.M

    def read(self, w):
        """ Read memory corresponding to the address weighting
        Arguments:
            w: shape = (batch_size,N)
        Outputs:
            shape = (batch_size,M)
        """
        return torch.matmul(w.unsqueeze(1), self.memory).squeeze(1)

    def write(self, w, e, a):
        """ Erase/Add memory corresponding to the address weighting
        Arguments:
            w: shape = (batch_size,N)
            e,a: shape = (batch_size,M)
        """
        self.prev_mem = self.memory
        erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(1))
        add = torch.matmul(w.unsqueeze(-1), a.unsqueeze(1))
        self.memory = self.prev_mem * (1 - erase) + add

    def address(self, k, beta, g, s, gamma, w_prev):
        """Address Mechanism"""
        # Content focus
        wc = self._similarity(k, beta)

        # Location focus
        wg = self._interpolate(w_prev, wc, g)
        w_hat = self._shift(wg, s)
        w = self._sharpen(w_hat, gamma)

        return w

    def _similarity(self, k, beta):
        """Content Addressing Mechanism"""

        cos_sim = F.cosine_similarity(self.memory + 1e-16, k.unsqueeze(1) + 1e-16, dim=-1)
        w = F.softmax(beta * cos_sim , dim=1)
        return w


    def _interpolate(self, w_prev, wc, g):
        """Interpolation Gate, blend weighting with last time-step"""

        return g * wc + (1 - g) * w_prev

    def _shift(self, wg, s):
        """Shifting Weighting, normalized distribution over allowed shifts"""

        # conved = torch.cat([wg[:,-1:],wg,wg[:,:1]],dim=1)
        # result = F.conv1d(conved.unsqueeze(1),s.unsqueeze(1))
        #FIXME: building time is pretty long if batch_size is large, other solutions?
        # return torch.cat([result[i:i+1,i,:] for i in range(self.batch_size)])

        result = torch.Tensor(self.batch_size,self.N)
        for b in range(self.batch_size):
            result[b] = _convolve(wg[b],s[b])
        return result

    def _sharpen(self, w_hat, gamma):
        """Sharpen final weighting, prevent blurring"""

        w = w_hat ** gamma
        w = torch.div(w, torch.sum(w, dim=1).view(-1, 1) + 1e-16)
        return w
