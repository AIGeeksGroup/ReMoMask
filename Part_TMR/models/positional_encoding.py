"""
# This code is based on https://github.com/Mathux/TMR
"""

import numpy as np
import torch
from torch import nn
import ipdb

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super().__init__()

        self.batch_first = batch_first  
        # If True, input shape is (batch_size, seq_len, d_model)
        # If False, input shape is (seq_len, batch_size, d_model)

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # torch.Size([5000, 512])
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # torch.Size([5000, 1])

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)       
        pe[:, 1::2] = torch.cos(position * div_term)        
        pe = pe.unsqueeze(0).transpose(0, 1)   # torch.Size([5000, 1, 512])  # (seq_len, batch_size, d_model)
        # pe.unsqueeze(0) : torch.Size([1, 5000, 512])
        # pe.unsqueeze(0).transpose(0, 1) : torch.Size([5000, 1, 512]) 

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x.shape: torch.Size([225, 1, 512])  # (seq_len, batch_size, d_model)
        # pe.shape: torch.Size([5000, 1, 512])  # (max_seq_len, batch_size, d_model)   # 位置编码图
        # not used in the final model
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        else:
            # x = x + self.pe[: x.shape[0], :]  torch.Size([225, 1, 512])  # (seq_len, batch_size, d_model)
            x = x + self.pe[: x.shape[0], :]  # torch.Size([225, 1, 512])  # (seq_len, batch_size, d_model)
        return self.dropout(x) # torch.Size([225, 1, 512])  # (seq_len, batch_size, d_model)
