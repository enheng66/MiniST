import sys
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from src.base.model import BaseModel



class Model(BaseModel):
    def __init__(self, configs,**args):
        super(Model, self).__init__(**args)
        self.seq_len = configs['seq_len']
        self.pred_len = configs['horizon']

        self.patch_len = configs['patch_len']
        self.basis_num=configs['basis_num']

        self.K = configs['K']
        self.is_normal = configs['is_normal']
        self.n_layer = configs['n_layer']
        self.period=configs['period']
        self.seq_dim=self.node_num
        self.seq_len2=self.period*self.basis_num
        self.pad_seq_len = 0

        self.n_x= self.seq_len//self.period
        if self.seq_len > self.n_x*self.period:
            self.pad_seq_len = (self.n_x+1)*self.period - self.seq_len
            self.n_x += 1
        self.tobasis = nn.Linear(self.n_x, self.basis_num)


        self.temp_dim_tid =32
        self.temp_dim_diw =32
        self.node_dim = 64
        self.time_of_day_size =configs['time_of_day_size']
        self.day_of_week_size = configs['day_of_week_size']
        self.flag = True if self.seq_len == 96  else False
        self.hidden_dim = self.flag*(self.node_dim + self.temp_dim_tid +self.temp_dim_diw)  + self.seq_len2
        
        if self.flag:
            self.time_in_day_emb = nn.Parameter(torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
            self.day_in_week_emb = nn.Parameter(torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)
            self.node_emb = nn.Parameter(torch.empty(self.node_num, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)

        self.aggregation = Aggregation(
            self.seq_len2, self.seq_dim, self.patch_len)

        self.blocks = nn.ModuleList()  # 使用 ModuleList 存储
        for i in range(self.n_layer):
            if i == self.n_layer - 1:
                pred_len_ = self.pred_len
            else:
                pred_len_ = self.hidden_dim
            block = Block(
                self.hidden_dim, pred_len_, self.seq_dim,
                self.patch_len,self.K)
            self.blocks.append(block)  # 添加到 ModuleList

        self.pro=nn.Linear(self.pred_len, self.pred_len)

    def forward(self, x,label=None):
        # x = [batch_size, seq_dim, seq_len, ]
        # y = [batch_size, seq_dim, pred_len, ]
        input=x
        b,t,n,_ = input.shape
        x = x[:, :, :,0]
        x = x.permute(0, 2, 1)
        if self.flag:
            t_i_d_data = input[..., 1]
            time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)]
            d_i_w_data = input[..., 2]
            day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :] * self.day_of_week_size).type(torch.LongTensor)]
            node_emb=self.node_emb.unsqueeze(0).expand(b, -1, -1)

        if self.is_normal:
            x_ = x.detach()
            x_mu = torch.mean(x_, 2, keepdim=True)
            x_sigma = torch.std(x_, 2, keepdim=True)
            x_sigma[x_sigma < 1e-6] = 1.0
            x = (x - x_mu) / x_sigma

        #basis
        if self.pad_seq_len>0:#、
            t = (self.n_x-1)*self.period
            x = torch.cat([x,x[:,:,t-self.pad_seq_len:t]],dim=-1)
            
        
        x = x.reshape(b,n,self.n_x,self.period)
        
        x = x.permute(0,1,3,2)
        x = x.reshape(-1,self.period,self.n_x)
        basis = self.tobasis(x)
        orthogonal_loss = cal_orthogonal_loss(basis)

        x=basis.reshape(b,n,self.period,self.basis_num).permute(0,1,3,2)
        x = x.reshape(b,n,-1)
        #agg
        x = self.aggregation(x)
        if self.flag:
            x=torch.cat([x, time_in_day_emb, day_in_week_emb, node_emb], dim=-1)

        #backnone
        for block in self.blocks:  # 仍然手动逐层计算
            x= block(x)
        
        x=self.pro(x)
        if self.is_normal:
            y = (x * x_sigma) + x_mu
            
        y = y.permute(0, 2, 1)
        y = y.unsqueeze(-1)
        return y, orthogonal_loss

    def get_n_param(self):
        n_param = 0
        for param in self.parameters():
            if param.requires_grad:
                n_param += torch.numel(param)
        return n_param

class Aggregation(nn.Module):
    def __init__(self, seq_len, seq_dim, patch_len):
        super(Aggregation, self).__init__()

        padding = int(patch_len // 2)
        kernel_size = int(1 + 2 * padding)
        self.conv1d = nn.Conv1d(
            in_channels=1, out_channels=1, 
            kernel_size=kernel_size,
            stride=1, padding=padding,
            padding_mode="zeros", bias=False)

        self.seq_dim = seq_dim
        self.seq_len = seq_len

    def forward(self, x):
        h = x.reshape(-1, 1, self.seq_len)
        h = self.conv1d(h)
        h = h.reshape(-1, self.seq_dim, self.seq_len)
        return h + x


