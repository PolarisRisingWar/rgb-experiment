import torch
from torch.functional import Tensor

def node_induced_subgraph(total_num_nodes,nodes_set,original_edge_index,reorder_nodes=True):
    #参考https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/subgraph.html#subgraph
    """
    入参：
    total_num_nodes是原始图的节点数
    node_set是节点索引的list或mask
    original_edge_index是原图的edge_index
    reorder_nodes：是否要对子图的节点重新从0开始索引。默认置True

    返回值：node-induced subgraph的edge_index
    """
    if isinstance(nodes_set,list):  #子图节点索引构成的list
        new_graph_num_node=len(nodes_set)
    elif isinstance(nodes_set,Tensor) and nodes_set.dtype==torch.bool:  #mask
        new_graph_num_node=sum(nodes_set).item()

    node_sampled_mask = torch.zeros(total_num_nodes, dtype=torch.bool)
    node_sampled_mask[nodes_set] = 1            
    mask = node_sampled_mask[original_edge_index[0]] & node_sampled_mask[original_edge_index[1]]
    #mask对应每一条边是否被选中
    edge_index = original_edge_index[:, mask]
    if reorder_nodes:
        n_idx = torch.zeros(total_num_nodes, dtype=torch.long)
        n_idx[nodes_set] = torch.arange(new_graph_num_node)
        edge_index = n_idx[edge_index]
    return edge_index