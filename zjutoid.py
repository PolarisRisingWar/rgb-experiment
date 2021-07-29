from typing import Optional, Callable, List

import os.path as osp
from collections import Counter
import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.utils import num_nodes
# from torch_geometric.io import read_planetoid_data
from iozjutoid import read_zjutoid_data,read_file,index_to_mask
from torch_geometric.data.makedirs import makedirs
import  numpy as np
from scipy.sparse import csr_matrix
from torch_sparse.tensor import  SparseTensor

import random

class zjutoid(InMemoryDataset):
    # url = 'https://github.com/kimiyoung/planetoid/raw/master/data'
    url ='https://github.com/YiQing12345/Toid-/raw/master/data'
    def __init__(self, root: str, name: str, split: str = "public",
                 ratio: str = "6-1-3",ratio_random:bool=False,
                 fixed_split_id:int=0,fixed_split_path:str="",
                 weight_ratio:str='None',
                 num_train_per_class: int = 20, num_val: int = 500,
                 num_test: int = 1000, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 seed:Optional[int]=0):
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.split = split
        assert self.split in ['public', 'full', 'random','ratio','fixed']
        self.node_num=len(self.data.test_mask)
        if split == 'full':
            data = self.get(0)
            data.train_mask.fill_(True)
            data.train_mask[data.val_mask | data.test_mask] = False
            self.data, self.slices = self.collate([data])

        elif split == 'random':
            data = self.get(0)
            data.train_mask.fill_(False)
            for c in range(self.num_classes):
                idx = (data.y == c).nonzero(as_tuple=False).view(-1)
                idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
                data.train_mask[idx] = True

            remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            data.val_mask.fill_(False)
            data.val_mask[remaining[:num_val]] = True

            data.test_mask.fill_(False)
            data.test_mask[remaining[num_val:num_val + num_test]] = True
            self.data, self.slices = self.collate([data])
        elif split == 'ratio':
            data = self.get(0)
            train_val_test_list=[int(i) for i in ratio.split('-')]
            #TODO:float形式
            print(train_val_test_list)
            while True:
                if not ratio_random:
                    random.seed(seed)
                tvt_sum=sum(train_val_test_list)
                tvt_ratio_list=[i/tvt_sum for i in train_val_test_list]
                num_nodes=self.node_num
                train_end_index=int(tvt_ratio_list[0]*num_nodes)
                val_end_index=train_end_index+int(tvt_ratio_list[1]*num_nodes)
                
                before_shuffle=list(range(num_nodes))
                random.shuffle(before_shuffle)
                
                train_mask_index=before_shuffle[:train_end_index]
                val_mask_index=before_shuffle[train_end_index:val_end_index]
                test_mask_index=before_shuffle[val_end_index:]
                
                train_mask=torch.tensor([False for i in range(num_nodes)])
                train_mask[train_mask_index]=True
                val_mask=torch.tensor([False for i in range(num_nodes)])
                val_mask[val_mask_index]=True
                test_mask=torch.tensor([False for i in range(num_nodes)])
                test_mask[test_mask_index]=True
                data.train_mask =train_mask
                data.test_mask =val_mask
                data.val_mask =test_mask
                if self.check_train_data(data):
                    break
                seed+=1
            print("seed:"+str(seed))
            self.data, self.slices = self.collate([data])
        elif split == 'fixed':
            if self.name == 'Yelpchi':
                self.fixed_split_path=fixed_split_path
                if not osp.exists(self.fixed_split_path):
                    print("fixed_split_path:"+str(self.fixed_split_path)+" doesn't exists!")
                else:
                    splits_list=np.load(self.fixed_split_path, allow_pickle=True)
                    if fixed_split_id>=len(splits_list):
                        print("The fixed_split_id is "+str(fixed_split_id)+", but yelp-chi-splits.npy only have "+str(len(splits_list))+" splits.")
                        return
                    split_dict=splits_list[fixed_split_id]
                    data=self.get(0)
                    data.train_mask=index_to_mask(split_dict['train'],size=data.y.size(0))
                    data.val_mask=index_to_mask(split_dict['valid'],size=data.y.size(0))
                    data.test_mask=index_to_mask(split_dict['test'],size=data.y.size(0))
                    self.data, self.slices = self.collate([data])
            else:
                print("The "+str(self.name)+" can't use the fixed split!")
        self.weight_ratio=weight_ratio
        self.clean_unknown_mask()
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        return [f'ind.{self.name.lower()}.{name}' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
    @property
    def weight(self) ->torch.tensor:
        if self.weight_ratio=='None':
            return torch.tensor([1 for i in range(self.num_classes)],dtype=torch.float)
        else:
            weight_list = [float(i) for i in self.weight_ratio.split('-')]
            return torch.tensor(weight_list,dtype=torch.float)
    # def calculate(self,the_y):
    #     the_y=the_y.numpy().tolist()
    #     print(Counter(the_y))
    def check_train_data(self,the_data):
        the_train_mask=the_data.train_mask.numpy()
        the_train_y=the_data.y[the_data.train_mask]
        #
        # print("the_train_y")
        # print(the_train_y)
        # print("train,test,val")
        # self.calculate(data.y[data.train_mask])
        # self.calculate(data.y[data.test_mask])
        # self.calculate(data.y[data.val_mask])
        if self.name=='Elliptic':
            num_classes=2
        else:
            num_classes=self.num_classes
        for c in range(num_classes):
            if c not in the_train_y:
                return False
        print("check_train_data yes!")
        return True
    def clean_unknown_mask(self):
        if self.name != 'Elliptic':
            return
        data = self.data
        # print("152 train,test,val")
        # self.calculate(data.y[data.train_mask])
        # self.calculate(data.y[data.test_mask])
        # self.calculate(data.y[data.val_mask])
        unknown=(data.y!=2)
        # print("unknown type:"+str(type(unknown)))
        # print(data.y[unknown])
        train_mask=[data.train_mask[i] and unknown[i] for i in range(len(data.train_mask))]
        test_mask=[data.test_mask[i] and unknown[i] for i in range(len(data.test_mask))]
        val_mask=[data.val_mask[i] and unknown[i] for i in range(len(data.val_mask))]
        data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
        data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
        # print("168 train,test,val")
        # self.calculate(data.y[data.train_mask])
        # self.calculate(data.y[data.test_mask])
        # self.calculate(data.y[data.val_mask])
        self.data, self.slices = self.collate([data])
        # self.data, self.slices = self.collate([data])
        # print(unknown)


    def download(self):
        for name in self.raw_file_names:
            download_url('{}/{}'.format(self.url, name), self.raw_dir)

    def process(self):
        data = read_zjutoid_data(self.raw_dir, self.name)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'

    def _download(self):
        if files_exist(self.raw_paths):  # pragma: no cover
            return
        print(self.raw_dir)
        makedirs(self.raw_dir)
        # self.download()
    def read_graph(self):
        # graph=read_file(self.raw_dir, self.name,'graph')
        # adj=np.zeros((self.node_num,self.node_num))
        # for start,to_list in graph.items():
        #     for to in to_list:
        #         adj[start][to]=1
        # adj=csr_matrix(adj)
        the_SparseTensor=SparseTensor.from_edge_index(edge_index=self.data.edge_index,sparse_sizes=(self.node_num,self.node_num))
        return the_SparseTensor

def files_exist(files: List[str]) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])