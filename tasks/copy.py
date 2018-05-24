from .base import TaskBase, _clip_grads,_report
from ntm.ntm import NTM
from tqdm import tqdm
import numpy as np
import torch
from torch import optim, nn
import random

class TaskCopy(TaskBase):
    def __init__(self,cfg):

        # Register Task
        super(TaskCopy,self).__init__('copy')
        self.seq_width = cfg['seq_width'] or 8
        inp_dim = self.seq_width
        outp_dim = self.seq_width
        ctrl_size = cfg['ctrl_size'] or 100 
        ctrl_num_layers = cfg['ctrl_num_layers'] or 1
        N = cfg['mem_size'] or 128
        M = cfg['mem_dim'] or 20
        num_heads = cfg['num_heads'] or 1

        self.num_batches = cfg['batch'] or 200
        self.batch_size = cfg['batch_size'] or 1 
        self.epoch = cfg['epoch'] or 1000

        self.net= NTM(inp_dim,outp_dim,ctrl_size,ctrl_num_layers,N,M,num_heads)
        self.optimizer = optim.RMSprop(self.net.parameters())
                             
    def train(self):

        for e in range(self.epoch):
            loss_ls = []
            cost_ls = []
            print("Epoch {}\n".format(e))
            for X,y in tqdm(self._data_gen):
                loss,cost = self._train_batch(X,y)
                loss_ls.append(loss.data.numpy())
                cost_ls.append(cost.data.numpy())
            print(np.average(np.array(loss_ls)))
            print(np.average(np.array(cost_ls)))
                # _report(loss,cost)
        print("[INFO] Training finished!")
    
    def _train_batch(self,X,y):
        self.optimizer.zero_grad()
        inp_seqlen = X.size(0)
        outp_seqlen = y.size(0)
        pred = torch.Tensor(y.size())
        

        self.net.init(self.batch_size)

        for i in range(inp_seqlen):
            self.net(X[i])

        ## Retrieve the output
        for o in range(outp_seqlen):
            # Feed dummy input
            pred[o],_ = self.net(torch.zeros(self.batch_size,self.seq_width))

        criterion = nn.BCELoss()

        loss = criterion(pred,y)
        loss.backward()
        _clip_grads(self.net)
        self.optimizer.step()

        # pred_out = pred.clone()
        # pred_out.round_()

        cost = torch.sum(torch.abs(torch.round(pred)-y))

        return loss,cost/self.batch_size

    @property
    def _data_gen(self,seq_len=None):
        for batch in range(self.num_batches):
            if seq_len is None:
                seq_len = random.randint(5,20)
            data = np.random.binomial(1,0.5,(seq_len+1,self.batch_size,self.seq_width))
            data = torch.Tensor(data)
            y = data.clone()
            data[seq_len,:,:] = 1.0 # delimiter

            yield data,y
