import torch
from torch import nn
import numpy as np

class LSTMController(nn.Module):
    def __init__(self,inp_dim,outp_dim,num_layers):
        super(LSTMController,self).__init__()
        self.inp_dim = inp_dim
        self.outp_dim = outp_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size = inp_dim,
            hidden_size = outp_dim,
            num_layers = num_layers,
        )

        self.lstm_h = nn.Parameter(torch.randn(num_layers, 1, outp_dim) * 0.05)
        self.lstm_c = nn.Parameter(torch.randn(num_layers, 1, outp_dim) * 0.05)

        self.reset_parameters()

    def create_new_state(self, batch_size):
        batched_lstm_h = self.lstm_h.clone().repeat(1, batch_size, 1)
        batched_lstm_c = self.lstm_c.clone().repeat(1, batch_size, 1)
        return batched_lstm_h, batched_lstm_c

    def reset_parameters(self):
        for p in self.lstm.parameters():
            if p.dim() == 1:
                nn.init.constant_(p, .0)
            else:
                nn.init.xavier_normal_(p,gain=1.4)

    @property
    def size(self):
        return self.inp_dim, self.outp_dim

    def forward(self, inp, prev_state):
        """
        inp.shape = (seq_len, batch, inp_dim)
        //Assume uni-directional
        c.shape = h.shape = (num_layer, batch, hid_dim)
        outp.shape = (seq_len, batch, hid_dim)
        """
        # x = x.unsqueeze(0)
        outp, state = self.lstm(inp, prev_state)
        return outp,state
        # return outp.squeeze(0), state

# class FNNController(nn.module):
