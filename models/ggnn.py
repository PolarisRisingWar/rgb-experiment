import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv

class GGNN(nn.Module):
    """
    model_init_param需要num_layers,hidden_unit,dropout_rate
    model_forward_param需要x,edge_index
    模型结构：线性网络+BN+dropout，num_layers层GatedGraphConv，BN+dropout，线性网络+BN+dropout
    hidden_unit是state_dim。annotation_dim我这里用的是跟state_dim（hidden_dim）等长
    """
    def __init__(self,num_layers,hidden_unit,input_dim,output_dim,dropout_rate):
        super(GGNN,self).__init__()

        self.num_layers=num_layers
        self.dropout_rate=dropout_rate

        self.lin1=nn.Linear(input_dim,hidden_unit)
        self.bn1=nn.BatchNorm1d(hidden_unit)
        self.conv=GatedGraphConv(hidden_unit,num_layers)
        self.bn2=nn.BatchNorm1d(hidden_unit)
        self.lin2=nn.Linear(hidden_unit,output_dim)
    
    def forward(self,x,edge_index):
        x=self.lin1(x)
        x=self.bn1(x)
        #x=F.relu(x)
        x=F.dropout(x,self.dropout_rate,self.training)
        x=self.conv(x,edge_index)
        x=self.bn2(x)
        #x=F.relu(x)
        x=F.dropout(x,self.dropout_rate,self.training)
        x=self.lin2(x)
        
        return {'out':F.log_softmax(x, dim=1),'emb':x}