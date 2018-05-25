import os
import random
import numpy as np
import torch
from torch import optim, nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ntm.ntm import NTM


class TaskBase(object):

    def __init__(self,name,mark):
        self.name = name
        self.mark = mark
        self.writer = SummaryWriter()

    def train(self):
        raise NotImplementedError

    def clip_grads(self,model):
        """Gradient clipping to the range [10, 10]."""
        params = list(filter(lambda p: p.grad is not None, model.parameters()))
        for p in params:
            p.grad.clamp_(-10, 10)

    def init(self):
        path = 'log/{}/{}'.format(self.name,self.mark)
        if not os.path.exists(path):
            os.makedirs(path)

    def report(self,loss,cost,idx):
        self.writer.add_scalar('loss',loss,idx)
        self.writer.add_scalar('cost',cost,idx)

    def draw_sample(self,pred,y,idx):
        seq_len = y.shape[-1]
        fig = torch.cat([torch.round(pred),torch.ones(1,1,1,seq_len),y],dim=2)
        self.writer.add_image('Posterior',pred,idx)
        self.writer.add_image('Prediction',fig,idx)

    def save_model(self,model,idx):
        path = "log/{}/{}/{}.model".format(self.name,self.mark,idx)
        torch.save(model.state_dict(),path)
