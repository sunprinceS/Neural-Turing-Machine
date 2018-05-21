import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class NTMMemory(nn.Module):
    def __init__(self, N, M):
        super(NTMMemory, self).__init__()

        self.N = N
        self.M = M

        # Not train mem_init
        self.register_buffer('mem_init', torch.Tensor(N, M))

        # Initialize memory bias
        stdev = 1 / (np.sqrt(N + M))
        nn.init.uniform(self.mem_init, -stdev, stdev)

    def create_new_state(self, batch_size):
        self.batch_size = batch_size
        self.memory = self.mem_init.clone().repeat(batch_size, 1, 1)

    @property
    def size(self):
        return self.N, self.M

    def read(self, w):
        """
        w.shape = (batch_size,N)
        memory.shape = (batch_size,N,M)
        outp.shape = (batch_size,1,M)
        """
        return torch.matmul(w.unsqueeze(1), self.memory)

    def write(self, w, e, a):
        """
        w.shape = (batch_size,N)
        memory.shape = (batch_size,N,M)
        e.shape = a.shape = (batch_size,M)
        erase.shape = add.shape = (batch_size,N,M)
        """
        self.prev_mem = self.memory
        erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(1))
        add = torch.matmul(w.unsqueeze(-1), a.unsqueeze(1))
        self.memory = self.prev_mem * (1 - erase) + add

    def address(self, k, beta, g, s, gamma, w_prev):
        # Content focus
        wc = self._similarity(k, beta)

        # Location focus
        wg = self._interpolate(w_prev, wc, g)
        w_hat = self._shift(wg, s)
        w = self._sharpen(w_hat, gamma)

        return w

    def _similarity(self, k, beta):
        """
        cos_sim.shape = (batch_size,N)
        """
        cos_sim = F.cosine_similarity(self.memory + 1e-16, k.unsqueeze(1) + 1e-16, dim=-1)
        w = F.softmax(beta * cos_sim , dim=1)
        return w

    def _interpolate(self, w_prev, wc, g):
        return g * wc + (1 - g) * w_prev

    def _shift(self, wg, s):
        conved = torch.cat([wg[:,-1:],wg,wg[:,:1]],dim=1)
        result = F.conv1d(conved.unsqueeze(1),s.unsqueeze(1))
        #FIXME: building time is pretty long if batch_size is large
        #TODO: abother elegant solution?
        return torch.cat([result[i:i+1,i,:] for i in range(self.batch_size)])

    def _sharpen(self, w_hat, gamma):
        w = w_hat ** gamma
        w = torch.div(w, torch.sum(w, dim=1).view(-1, 1) + 1e-16)
        return w
