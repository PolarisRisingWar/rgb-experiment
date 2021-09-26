#参考https://github.com/pyg-team/pytorch_geometric/blob/master/examples/super_gat.py

import torch
import torch.nn.functional as F

from torch_geometric.nn import SuperGATConv

class SuperGAT(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,heads,dropout_rate,
                edge_sample_ratio:1,neg_sample_ratio:0.5):
        super(SuperGAT, self).__init__()

        self.dropout_rate=dropout_rate

        self.conv1 = SuperGATConv(input_dim,hidden_dim, heads=heads,
                                  dropout=dropout_rate,edge_sample_ratio=edge_sample_ratio,
                                  neg_sample_ratio=neg_sample_ratio)
        self.conv2 = SuperGATConv(hidden_dim*heads, output_dim, heads=heads,
                                  concat=False, dropout=dropout_rate,
                                  edge_sample_ratio=edge_sample_ratio,
                                  neg_sample_ratio=neg_sample_ratio)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        att_loss = self.conv1.get_attention_loss()
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        att_loss += self.conv2.get_attention_loss()
        return {'out':F.log_softmax(x, dim=-1),'att_loss':att_loss,'emb':x}