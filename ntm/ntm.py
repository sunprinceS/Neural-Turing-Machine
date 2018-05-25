import torch
from torch import nn
import torch.nn.functional as F

from .controller import LSTMController
from .head import NTMReadHead,NTMWriteHead
from .memory import NTMMemory

class NTM(nn.Module):
    def __init__(self,inp_dim,outp_dim,ctrl_size,ctrl_num_layers,N,M,num_heads):
        super(NTM,self).__init__()

        self.inp_dim = inp_dim
        self.outp_dim = outp_dim
        self.ctrl_size = ctrl_size
        self.ctrl_num_layers = ctrl_num_layers
        self.controller = LSTMController(inp_dim + num_heads * M, ctrl_size, ctrl_num_layers)
        self.memory = NTMMemory(N,M)
        self.heads = nn.ModuleList([])
        self.init_rs = []
        for i in range(num_heads):
            self.heads +=[
                NTMReadHead(self.memory,ctrl_size),
                NTMWriteHead(self.memory,ctrl_size)
            ]
            init_r = torch.Tensor(1,self.memory.size[1])
            self.register_buffer('init_r{}'.format(i),init_r)
            self.init_rs.append(init_r)
        self.fc = nn.Linear(num_heads * M, outp_dim)
        self.reset_parameters()
        
    def init(self,batch_size):
        self.batch_size = batch_size
        self.prev_state = self.create_new_state()

    def create_new_state(self):
        """Init new state and batchify"""

        self.memory.create_new_state(self.batch_size)
        init_rs = [r.clone().repeat(self.batch_size,1) for r in self.init_rs]
        ctrl_state = self.controller.create_new_state(self.batch_size)
        heads_state = [head.create_new_state(self.batch_size) for head in self.heads]

        return init_rs,ctrl_state,heads_state

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight,gain=1)
        nn.init.normal_(self.fc.bias,std=0.01)
        for init_r in self.init_rs:
            nn.init.normal_(init_r)

    def forward(self,x):
        prev_reads,prev_ctrl_state,prev_heads_state = self.prev_state
        inp = torch.cat([x] + prev_reads,dim=1)
        ctrl_outp,ctrl_state = self.controller(inp,prev_ctrl_state)
        reads = []
        heads_state = []
        for head,prev_head_state in zip(self.heads,prev_heads_state):
            if head.is_read_head():
                r,head_state = head(ctrl_outp,prev_head_state)
                reads += [r]
            else:
                head_state = head(ctrl_outp,prev_head_state)
            heads_state += [head_state]

        # Retrieve output according to current reads
        inp2 = torch.cat(reads, dim=1)
        o = F.sigmoid(self.fc(inp2)) # range: [0,1]
        self.prev_state = (reads,ctrl_state,heads_state)

        return o, self.prev_state
    
    def calculate_num_params(self):
        return sum([p.view(-1).size(0) for p in self.parameters()])
