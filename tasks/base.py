import os
import random
import torch
from torch import optim, nn
from torchvision import transforms
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from tqdm import tqdm
from ntm.ntm import NTM


class TaskBase(object):

    def __init__(self,name,mark,mode):
        self.name = name
        print(name)
        self.mark = mark
        self.mode = mode
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

    def report(self,loss,cost,test_cost,idx):
        self.writer.add_scalar('loss/loss',loss,idx)
        self.writer.add_scalars('cost',{'cost':cost,'val_cost':test_cost},idx)

    def draw_sample(self,pred,y,idx):
        seq_len = y.shape[-1]
        fig = torch.cat([torch.round(pred),torch.ones(1,1,1,seq_len),y],dim=2)
        self.writer.add_image('L{}/Posterior'.format(seq_len),pred,idx)
        self.writer.add_image('L{}/Prediction'.format(seq_len),fig,idx)

    def draw_memory(self,seq_len,idx):
        memory = self.net.memory
        N,M = memory.size
        # exit()
        # memory = F.sigmoid(memory.get_memory().t().view(1,M,N))
        memory = memory.get_memory().t().view(1,M,N)
        enlarge = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((120,840)),
            transforms.ToTensor()])
        memory = enlarge(memory)

        self.writer.add_image('L{}/Memory'.format(seq_len),memory.unsqueeze(0),idx)
        
    def save_model(self,model,idx):
        path = "log/{}/{}/{}.model".format(self.name,self.mark,idx)
        torch.save(model.state_dict(),path)
