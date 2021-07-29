import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import APPNP

class APPNPStack(nn.Module):
    """
    model_init_param需要hidden_unit,K,alpha,dropout_rate
    model_forward_param需要x,edge_index
    模型结构：1层线性网络，1层(APPNP+BN+dropout)，1层线性网络，1层dropout
    """
    def __init__(self,hidden_unit,input_dim,output_dim,K,alpha,dropout_rate):
        super(APPNPStack,self).__init__()

        self.dropout_rate=dropout_rate

        self.lin1=nn.Linear(input_dim,hidden_unit)
        self.conv=APPNP(K,alpha)
        self.bn=nn.BatchNorm1d(hidden_unit)
        self.lin2=nn.Linear(hidden_unit,output_dim)
        
    
    def forward(self,x,edge_index):
        x=self.lin1(x)
        x=self.conv(x,edge_index)
        x=self.bn(x)
        x=F.dropout(x,p=self.dropout_rate,training=self.training)
        x=self.lin2(x)
        #x=F.relu(x)
        #x=F.dropout(x,p=self.dropout_rate,training=self.training)
        
        return {'out':F.log_softmax(x, dim=1),'emb':x}