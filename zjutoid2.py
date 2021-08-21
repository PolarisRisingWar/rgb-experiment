#数据文件夹中的文件：torch.Tensor格式的数据
#y.pt：分类任务：-1-(num_class-1)的数字，-1表示该节点无标签 (转换为LongTensor)
#x.pt
#edge_index.pt（不限制必须是LongTensor，但是我写的都是的）

import datetime

import torch
from torch.functional import Tensor

from torch_geometric.data import Data

import random

#函数部分
class zjutoid2():
    def __init__(self,dataset_name:str,dataset_root:str,
            split_method:str='ratio',split_seed=1234567,
            split_ratio:str='6-2-2',num_train_per_class:int=20,num_val:int=500,
            num_test:int=1000,
            apply_sample:bool=False,sample_seed=1234567,
            sample_method:str='random',sample_criterion='node',sample_count_method='ratio',
            sample_num=1000,sample_ratio=0.8,
            sample_rw_length=None,
            remove_non_label_node:bool=False,specify_non_label_mask:bool=False):
        """
        入参：
        dataset_name要求与数据文件夹同名
        dataset_root内置数据文件夹，数据文件夹下储存原始数据文件
        TODO：如果没有，就自己创造文件夹，并使用download()函数下载数据
        split_method为'ratio'时，按照split_ratio的比例划分数据集
        TODO：split_method为'default'时，返回内置的数据集mask（但是要抽样啊啥的之后又咋整？）
        TODO：split_method为'random'时，参考Planetoid类的数据集划分方式
            使用入参num_train_per_class num_val num_test进行划分（如果超过了节点数需要判断）
        apply_sample：是否应用图采样技术
        sample_method为'random'时
            如果sample_criterion为'node'，随机抽样节点，返回node-induced subgraph
            TODO:找点更好的node-induced subgraph方法吧，遍历总觉得不太对劲
            TODO：如果sample_criterion为'edge'，随机抽样边，返回edge-induced subgraph
            sample_count_method='ratio' (sample_ratio) / TODO: 'num' (sample_num)（判断数量）
            （注意这里的ratio，如果要移除无标签节点，是移除之后的ratio）
        TODO:sample_method为'SRW'时，应用simple random walk采样
            随机抽取一个起始节点模拟random walk（长度为sample_rw_length）
            返回节点序列的node-induced subgraph
        remove_non_label_node：如果置True则移除无标签节点，即返回有标签节点及其node-induced subgrah
        specuft_non_label_mask：在数据中设置non_label_mask（无标签节点为True）


        self.data：PyG的Data格式图数据
        属性：x, y, edge_index, train_mask, val_mask, test_mask, non_label_mask (optional)
        """
        x=torch.load(dataset_root+'/'+dataset_name+'/x.pt')
        y=torch.load(dataset_root+'/'+dataset_name+'/y.pt')
        y=y.long()
        edge_index=torch.load(dataset_root+'/'+dataset_name+'/edge_index.pt')
        edge_index=edge_index.long()

        self.num_nodes=x.size()[0]
        self.split_seed=split_seed

        if remove_non_label_node:
            have_label_mask=y!=-1
            x=x[have_label_mask]
            y=y[have_label_mask]
            edge_index=self.node_induced_subgraph(have_label_mask,edge_index)
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
                        edge_index=self.node_induced_subgraph(sample_nodes,edge_index)

                        self.num_nodes=after_sample_num_nodes
        
        

        data=Data(x=x,y=y,edge_index=edge_index)

        #配置non_label_mask
        self.non_label_mask=y==-1
        #print(self.non_label_mask.sum())
        if specify_non_label_mask:
            data.non_label_mask=self.non_label_mask

        #配置train_val_test mask：注意没有标签的节点不参与配置
        if split_method=='ratio':
            self.remake_mask(data,split_ratio)
        
        self.data=data



    def create_dir(self):
        pass

    def download_data(self):
        pass



    def make_mask(self,ratio):
        have_label_mask=~self.non_label_mask
        
        bs=torch.tensor(list(range(self.num_nodes)),dtype=int)
        bs_label=bs[have_label_mask]
        num_have_label=len(bs_label)
        bs_label_index=list(range(num_have_label))
        random.shuffle(bs_label_index)

        train_val_test_list=[int(i) for i in ratio.split('-')]
        random.seed(self.split_seed)
        tvt_sum=sum(train_val_test_list)
        tvt_ratio_list=[i/tvt_sum for i in train_val_test_list]
        train_end_index=int(tvt_ratio_list[0]*num_have_label)
        val_end_index=train_end_index+int(tvt_ratio_list[1]*num_have_label)
        
        train_mask_index=bs_label_index[:train_end_index]
        val_mask_index=bs_label_index[train_end_index:val_end_index]
        test_mask_index=bs_label_index[val_end_index:]
        
        train_mask=torch.zeros(self.num_nodes,dtype=torch.bool)
        train_mask[bs_label[train_mask_index]]=True
        val_mask=torch.zeros(self.num_nodes,dtype=torch.bool)
        val_mask[bs_label[val_mask_index]]=True
        test_mask=torch.zeros(self.num_nodes,dtype=torch.bool)
        test_mask[bs_label[test_mask_index]]=True

        return (train_mask,val_mask,test_mask)

    def check_train_containing(self,train_mask,y):
        """（仅用于分类任务）检查train_mask中是否含有y中所有的标签（-1不算）"""
        for label in y.unique():
            l=label.item()
            if l==-1:
                continue
            if l not in y[train_mask]:
                return False
        return True

    def remake_mask(self,data,ratio):
        """直接覆盖原data的train_mask, val_mask, test_mask三个属性"""
        while True:
            (train_mask,val_mask,test_mask)=self.make_mask(ratio)
            if self.check_train_containing(train_mask,data.y):
                data.train_mask=train_mask
                data.val_mask=val_mask
                data.test_mask=test_mask
                break
            else:
                self.split_seed+=1
    



    def node_induced_subgraph(self,nodes_set,original_edge_index,reorder_nodes=True):
        #参考https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/subgraph.html#subgraph
        """
        入参：node_set是节点索引的list或mask，original_edge_index是原图的edge_index
            reorder_nodes：是否要对子图的节点重新从0开始索引。默认置True
        返回值：node-induced subgraph的edge_index
        """
        if isinstance(nodes_set,list):  #list
            new_graph_num_node=len(nodes_set)
        elif isinstance(nodes_set,Tensor) and nodes_set.dtype==torch.bool:  #mask
            new_graph_num_node=sum(nodes_set).item()
            
        node_sampled_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        node_sampled_mask[nodes_set] = 1            
        mask = node_sampled_mask[original_edge_index[0]] & node_sampled_mask[original_edge_index[1]]
        #整数列表索引这部分知识点我确实运用不娴熟
        edge_index = original_edge_index[:, mask]
        if reorder_nodes:
            n_idx = torch.zeros(self.num_nodes, dtype=torch.long)
            n_idx[nodes_set] = torch.arange(new_graph_num_node)
            edge_index = n_idx[edge_index]
        return edge_index





#测试部分
#"""
z=zjutoid2('bgp','/data/wanghuijuan/dataset1/zjutoid2_ds',specify_non_label_mask=True,
        apply_sample=False,remove_non_label_node=True)
print(z.data.is_directed())
print(z.data)
#print(z.data.edge_index.max())
#"""