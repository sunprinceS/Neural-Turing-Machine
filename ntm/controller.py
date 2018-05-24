import torch
from torch import nn
import numpy as np

class LSTMController(nn.Module):
    def __init__(self,inp_dim,outp_dim,num_layers):
        super(LSTMController,self).__init__()
        self.inp_dim = inp_dim
        self.outp_dim = outp_dim
        self.num_layers = num_layers

        # Core
        self.lstm = nn.LSTM(
            input_size = inp_dim,
            hidden_size = outp_dim,
            num_layers = num_layers,
        )
        
        # set init state can be trained
        self.lstm_h = nn.Parameter(torch.randn(num_layers, 1, outp_dim) * 0.05)
        self.lstm_c = nn.Parameter(torch.randn(num_layers, 1, outp_dim) * 0.05)

        self.reset_parameters()

    def create_new_state(self, batch_size):
        """ Init new state and batchify
        Outputs:
            batch_state: shape = (num_layers,batch_size,ctrl_size)
        """
        batched_lstm_h = self.lstm_h.clone().repeat(1, batch_size, 1)
        batched_lstm_c = self.lstm_c.clone().repeat(1, batch_size, 1)
        return batched_lstm_h, batched_lstm_c

    def reset_parameters(self):
        for p in self.lstm.parameters():
            if p.dim() == 1:
                nn.init.constant_(p, .0)
            else:
                nn.init.xavier_normal_(p,gain=1.2)
                # stdev = 5 / (np.sqrt(self.inp_dim+  self.outp_dim))
                # nn.init.uniform(p, -stdev, stdev)

    @property
    def size(self):
        return self.inp_dim, self.outp_dim

    def forward(self, inp, prev_state):
        """
        Args:
            inp: input of controller (external input + read_head state)
                shape: (batch_size,inp_dim + num_read_head * mem_dim)
            prev_state: (h_{t-1},c_{t-1})
                shape: both (num_layers,batch_size,ctrl_size)
        Outputs:
            outp: will use as the embedding to generate related param, k, β, g, s, γ, (e, a)
                shape: (batch_size, ctrl_size)
            state: (h_t,c_t)
                shape: both (num_layers, batch_size, ctrl_size)
        """
        outp, state = self.lstm(inp.unsqueeze(0), prev_state)
        return outp.squeeze(0),state
