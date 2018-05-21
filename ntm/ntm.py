import torch
from torch import nn
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
        self.rhead_inits = nn.ModuleList([])
        for i in range(num_heads):
            self.heads +=[
                NTMReadHead(self.memory,ctrl_size),
                NTMWriteHead(self.memory,ctrl_size)
            ]
            rhead_init = torch.randn(1,self.M) * 0.01
            self.register_buffer('rhead{}_init'.format(i),self.rhead_init)
            self.rhead_inits += [rhead_init]
        self.fc = nn.Linear(ctrl_size + num_heads * M, outp_dim)
        self.reset_parameters()
        
    def create_new_state(self,batch_size):
        self.batch_size = batch_size
        rhead_inits = [r.clone().repeat(batch_size,1) for r in self.rhead_inits]
        ctrl_state = self.controller.create_new_state(batch_size)
        heads_state = [head.create_new_state(batch_size) for head in self.heads]
        self.memory.reset(batch_size)

        return rhead_inits,ctrl_state,heads_state

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight,gain=1.4)
        nn.init.normal_(self.fc.bias,std=0.01)

    def forward(self,x,prev_state):
        """
        x.shape = (batch_size,inp_dim)
        """
        prev_reads,prev_ctrl_state,prev_heads_state = prev_state
        inp = torch.cat([x] + prev_reads,dim=1)
        ctrl_outp,ctrl_state = self.controller(inp,prev_ctrl_state)

        reads = []
        heads_state = []
        for head,prev_heads_state in zip(self.heads,prev_heads_state):
            if head.is_read_head():
                r,head_state = head(controller_outp,prev_head_state)
                reads += [r]
            else:
                head_state = head(controller_outp,prev_head_state)
            heads_state += [head_state]

        inp2 = torch.cat([ctrl_outp] + reads, dim=1)
        o = F.sigmoid(self.fc(inp2))

        return o, (reads,ctrl_state,heads_states)
    
    def calculate_num_params(self):
        return torch.sum([p.view(-1).size(0) for p in self.parameters()])
