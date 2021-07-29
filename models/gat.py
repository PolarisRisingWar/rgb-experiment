import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv

class GAT(nn.Module):
    """
    model_init_param需要num_layers,hidden_unit,dropout_rate,heads
    model_forward_param需要x,edge_index
    模型结构：num_layers（最少为1）层(卷积网络+BN+dropout)+1层线性网络+dropout
    """
    def __init__(self,num_layers,hidden_unit,input_dim,output_dim,dropout_rate,heads):
        super(GAT,self).__init__()

        self.num_layers=num_layers
        self.dropout_rate=dropout_rate

        self.convs=nn.ModuleList()
        self.convs.append(GATConv(input_dim,hidden_unit,heads))
        for i in range(num_layers-1):
            self.convs.append(GATConv(hidden_unit*heads,hidden_unit,heads))
        self.bns=nn.ModuleList([nn.BatchNorm1d(hidden_unit*heads) for i in range(num_layers)])

        self.lin=nn.Linear(hidden_unit*heads,output_dim)
        #这个线性层是用来转换维度的，参考cs224w的实现。可参考：
        #https://blog.csdn.net/PolarisRisingWar/article/details/118545695
        #然后我看了一下PyG官方这个：
        #https://github.com/rusty1s/pytorch_geometric/blob/e6b8d6427ad930c6117298006d7eebea0a37ceac/examples/ogbn_products_gat.py
        #是叠了一层concat=False的GATConv
        #都行吧大概
    
    def forward(self,x,edge_index):
        for i in range(self.num_layers):
            x=self.convs[i](x,edge_index)
            x=self.bns[i](x)
            #x=F.relu(x)
            x=F.dropout(x,p=self.dropout_rate,training=self.training)
        x=self.lin(x)
        
        return {'out':F.log_softmax(x, dim=1),'emb':x}