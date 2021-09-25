#注意1：仅可用于multi-class雷的单图节点分类任务的数据集:
#注意2：标签为-1时认为是无标签节点
#注意3：在这一部分所使用的都是CPU，所以返回的数据如需放到GPU上需要手动再放

import numpy as np

import torch

from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops

from torch_sparse import coalesce

import random

from .utils import get_whole_mask,get_classification_mask,get_random_mask,node_induced_subgraph


class RD2PD():
    """
    raw data to pytorch_geometric data
    """

    def __init__(self,dataset_name:str,dataset_root:str,*,
            split_method:str='ratio',split_seed=1234567,split_ratio:str='6-2-2',
            num_train_per_class:int=20,num_val:int=500,num_test:int=1000,
            remove_duplicate_edges:bool=False,remove_self_loop:bool=False,
            remove_non_label_node:bool=False,specify_non_label_mask:bool=False,
            apply_sample:bool=False,sample_seed=1234567,
            sample_method:str='random',
            sample_criterion='node',
            sample_count_method='ratio',sample_num=1000,sample_ratio=0.8,
            sample_rw_length=None,):
        """
        入参：
        必选：
        dataset_name：数据集名称，要求与数据文件夹同名
        dataset_root：原始文件路径，内置数据文件夹，数据文件夹下储存原始数据文件
        TODO：如果没有，就自己创造文件夹，并使用download()函数下载数据

        可选：
        数据集划分部分：
        split_method
            'ratio'：按照split_ratio的比例对整体的数据集进行划分
            'classification':对数据集的每一类按split_ratio的比例进行划分
            TODO: 'default': 返回内置的数据集mask（使用原论文或其他重要论文所采用的数据划分方式）
                （禁止使用采样技术）
            'random': 参考Planetoid类的数据集划分方式
                使用入参num_train_per_class num_val num_test进行划分
                TODO: 处理总节点数小于输入数据集划分总节点数的情况
        
        数据集预处理部分：
        remove_duplicate_edges: 去除重边
        remove_self_loop：去除自环
        对无标签节点（以标签为-1来进行标识）的处理：
        remove_non_label_node：去除无标签节点
        specify_non_label_mask：在数据中设置non_label_mask（无标签节点为True）
        
        图采样部分：
        apply_sample：是否应用图采样技术
        sample_seed: 保持图抽样可复现性时使用的随机种子
        sample_method
            'random'
                sample_criterion
                    'node': 随机抽样节点，返回node-induced subgraph
                    TODO: 'edge'，随机抽样边，返回edge-induced subgraph
                sample_count_method
                    'ratio': 采样比例为sample_ratio的节点/边（如果要移除无标签节点，是移除之后的ratio）
                    TODO: 'num': 采样数量为sample_num的节点/边
            TODO:'SRW': 应用simple random walk采样
                随机抽取一个起始节点模拟random walk（长度为sample_rw_length）
                返回节点序列的node-induced subgraph

        
        self.data：PyG的Data格式图数据
        self.num_nodes: self.data的节点数
        属性：x, y, edge_index, train_mask, val_mask, test_mask, non_label_mask (optional)
        TODO:缓存功能以加速下载
        """
        #TODO：检测原始文件是否在路径中存在，如不存在则调用download_data()下载原始数据
        numpy_x=np.load(dataset_root+'/'+dataset_name+'/x.npy')
        x=torch.from_numpy(numpy_x).to(torch.float)
        numpy_y=np.load(dataset_root+'/'+dataset_name+'/y.npy')
        y=torch.from_numpy(numpy_y).to(torch.long)
        numpy_edge_index=np.load(dataset_root+'/'+dataset_name+'/edge_index.npy')
        edge_index=torch.from_numpy(numpy_edge_index).to(torch.long)

        if remove_duplicate_edges:
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))
        #看whj_code2/whj_dataset1/zjutoid2/wikipedianetwork.py的效果感觉是去掉重边的意思

        if remove_self_loop:
            edge_index, _ = remove_self_loops(edge_index)
        #参考自torch_geometric.datasets.Planetoid部分的代码

        self.num_nodes=x.size()[0]

        if remove_non_label_node:
            have_label_mask=y!=-1
            x=x[have_label_mask]
            y=y[have_label_mask]
            edge_index=node_induced_subgraph(self.num_nodes,have_label_mask,edge_index)
            self.num_nodes=x.size()[0]
        
        #图采样
        if apply_sample:
            random.seed(sample_seed)
            if sample_method=='random':
                if sample_criterion=='node':
                    if sample_count_method=='ratio':
                        before_sample_num_nodes=self.num_nodes
                        after_sample_num_nodes=int(before_sample_num_nodes*sample_ratio)
                        sample_nodes=random.sample(list(range(before_sample_num_nodes)),after_sample_num_nodes)
                        x=x[sample_nodes]
                        y=y[sample_nodes]
                        edge_index=node_induced_subgraph(self.num_nodes,sample_nodes,edge_index)

                        self.num_nodes=after_sample_num_nodes
        

        data=Data(x=x,y=y,edge_index=edge_index)

        #配置non_label_mask
        self.non_label_mask=y==-1
        #print(self.non_label_mask.sum())
        if specify_non_label_mask:
            data.non_label_mask=self.non_label_mask

        #数据集划分：配置train_val_test mask：注意没有标签的节点不参与配置
        assert split_method in ['ratio','classification','random']
        if split_method=='ratio':
            (train_mask,val_mask,test_mask)=get_whole_mask(y,split_ratio,split_seed)
        elif split_method=='classification':
            (train_mask,val_mask,test_mask)=get_classification_mask(y,split_ratio,split_seed)
        elif split_method=='random':
            (train_mask,val_mask,test_mask)=get_random_mask(y,num_train_per_class,num_val,num_test,split_seed)
        data.train_mask=train_mask
        data.val_mask=val_mask
        data.test_mask=test_mask
        

        self.data=data




    def download_data(self):
        """下载原始数据"""
        pass