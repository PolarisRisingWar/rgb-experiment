import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv

class GraphSAGE(nn.Module):
    """
    model_init_param需要num_layers,hidden_unit,dropout_rate
    model_forward_param需要x,edge_index
    模型结构：1层线性网络，num_layers层(卷积网络+BN+dropout)，1层线性网络
    """
    def __init__(self,num_layers,hidden_unit,input_dim,output_dim,dropout_rate):
        super(GraphSAGE,self).__init__()

        self.num_layers=num_layers
        self.dropout_rate=dropout_rate

        self.lin1=nn.Linear(input_dim,hidden_unit)
        #之所以非要叠一层是因为GitHub数据集转无向图不知为啥就OOM了，要不然谁乐意干这种事

        self.convs=nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(SAGEConv(hidden_unit,hidden_unit))

        self.bns=nn.ModuleList()
        for i in range(num_layers+1):
            self.bns.append(nn.BatchNorm1d(hidden_unit))
        
        self.lin2=nn.Linear(hidden_unit,output_dim)
    
    def forward(self,x,edge_index):
        x=self.lin1(x)
        x=self.bns[0](x)
        x=F.dropout(x,p=self.dropout_rate,training=self.training)
        for i in range(self.num_layers):
            x=self.convs[i](x,edge_index)
            x=self.bns[i+1](x)
            x=F.dropout(x,p=self.dropout_rate,training=self.training)
        x=self.lin2(x)
        
        return {'out':F.log_softmax(x, dim=1),'emb':x}