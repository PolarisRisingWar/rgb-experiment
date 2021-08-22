import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np

import math

#基本上完全照着PTA原项目来的，没有自己调结构

class Linear(nn.Module):
    """模型结构：dropout+线性转换（默认无偏置）"""
    def __init__(self, in_features, out_features, dropout, bias=False):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        input = F.dropout(input, self.dropout, training=self.training)
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output



class PTA(nn.Module): 
    """
    model_init_param需要nhid, dropout, epsilon, mode, K, alpha（没有给开PTS和PTD）
    神经网络部分结构：Linear+relu+Linear
    自定义损失函数
    inference() 函数是post-processing（就类似C&S那样的）
    TODO: 从PTA其实我可以得到灵感，就C&S感觉也可以这样每一epoch跟着一起post-ps啊，就可以输出
        每一轮的loss和ACC值了。以后可以写！
    """
    def __init__(self, nfeat, nhid, nclass, dropout, epsilon, K, alpha, mode=2):
        # mode: 0-PTS, 1-PTD, 2-PTA
        super(PTA, self).__init__()
        self.Linear1 = Linear(nfeat, nhid, dropout, bias=True)
        self.Linear2 = Linear(nhid, nclass, dropout, bias=True)
        self.epsilon = epsilon
        self.mode = mode
        self.K = K 
        self.alpha = alpha
        self.number_class = nclass

    def forward(self, x): 
        x = torch.relu(self.Linear1(x))
        return self.Linear2(x)

    def loss_function(self, y_hat, y_soft, epoch = 0): 
        if self.training: 
            y_hat_con = torch.detach(torch.softmax(y_hat, dim=-1))
            exp = np.log(epoch / self.epsilon + 1)
            if self.mode == 2: 
                loss = - torch.sum(torch.mul(torch.log_softmax(y_hat, dim=-1), torch.mul(y_soft, y_hat_con**exp))) / self.number_class  # PTA
            elif self.mode == 1:
                loss = - torch.sum(torch.mul(torch.log_softmax(y_hat, dim=-1), torch.mul(y_soft, y_hat_con))) / self.number_class  # PTD
            else: 
                loss = - torch.sum(torch.mul(torch.log_softmax(y_hat, dim=-1), y_soft)) / self.number_class # PTS
        else: 
            loss = - torch.sum(torch.mul(torch.log_softmax(y_hat, dim=-1), y_soft)) / self.number_class
        return loss

    def inference(self, h, adj): 
        y0 = torch.softmax(h, dim=-1) 
        y = y0
        for i in range(self.K):
            y = (1 - self.alpha) * torch.matmul(adj, y) + self.alpha * y0
        return y