import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv

class GCN(nn.Module):
    """
    model_init_param需要num_layers,hidden_unit,dropout_rate
    model_forward_param需要x,edge_index
    模型结构：num_layers（至少为2）层卷积网络
    """
    def __init__(self,num_layers,hidden_unit,input_dim,output_dim,dropout_rate):
        super(GCN,self).__init__()

        self.num_layers=num_layers
        self.dropout_rate=dropout_rate

        self.convs=nn.ModuleList()
        self.convs.append(GCNConv(input_dim,hidden_unit))
        for i in range(num_layers-2):
            self.convs.append(GCNConv(hidden_unit,hidden_unit))
        self.convs.append(GCNConv(hidden_unit,output_dim))
        #看官方实现：
        #https://github.com/rusty1s/pytorch_geometric/blob/e6b8d6427ad930c6117298006d7eebea0a37ceac/benchmark/kernel/gcn.py
        #这边和GAT的正好相反，绝了。
        #其他示例：
        #https://github.com/rusty1s/pytorch_geometric/blob/e6b8d6427ad930c6117298006d7eebea0a37ceac/benchmark/citation/gcn.py

        self.bn=nn.BatchNorm1d(output_dim)
    
    def forward(self,x,edge_index):
        for i in range(self.num_layers):
            x=self.convs[i](x,edge_index)
        #x=self.bn(x)
        #x=F.relu(x)
        #x=F.dropout(x,p=self.dropout_rate,training=self.training)
        
        return {'out':F.log_softmax(x, dim=1),'emb':x}