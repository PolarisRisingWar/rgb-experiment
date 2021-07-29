import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import APPNP

class APPNPStack(nn.Module):
    """
    model_init_param需要num_layers,hidden_unit,K,alpha,dropout_rate
    model_forward_param需要x,edge_index
    模型结构：1层线性网络，num_layers层(BN+APPNP+dropout)，1层线性网络，1层dropout
    """
    def __init__(self,num_layers,hidden_unit,input_dim,output_dim,K,alpha,dropout_rate):
        super(APPNPStack,self).__init__()

        self.num_layers=num_layers
        self.dropout_rate=dropout_rate

        self.lin1=nn.Linear(input_dim,hidden_unit)
        self.convs=nn.ModuleList([APPNP(K,alpha) for i in range(num_layers)])
        self.bns=nn.ModuleList([nn.BatchNorm1d(hidden_unit) for i in range(num_layers)])
        self.lin2=nn.Linear(hidden_unit,output_dim)
        
    
    def forward(self,x,edge_index):
        x=self.lin1(x)
        for i in range(self.num_layers):
            x=self.convs[i](x,edge_index)
            x=self.bns[i](x)
            #x=F.relu(x)
            x=F.dropout(x,p=self.dropout_rate,training=self.training)
        x=self.lin2(x)
        #x=F.relu(x)
        #x=F.dropout(x,p=self.dropout_rate,training=self.training)
        
        return {'out':F.log_softmax(x, dim=1),'emb':x}