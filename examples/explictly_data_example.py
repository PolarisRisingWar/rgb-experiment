#通过PyG.datasets直接获得Data格式数据（也可以通过其他方式获取数据）

import sys

import torch
sys.path.extend(['whj_code2/integration_experiment'])
from rgb_experiment import experiment

from torch_geometric.datasets import WebKB

#导入Data数据
ds_path='/data/wanghuijuan/dataset1/pyg_ds'
dataset=WebKB(ds_path,"Cornell")
print(dataset)
data=dataset.data
data.train_mask=data.train_mask[:,0]
data.val_mask=data.val_mask[:,0]
data.test_mask=data.test_mask[:,0]
#print(data.train_mask.size()==torch.Size([data.num_nodes]))
#print(len(data.train_mask.size()))
#print(data.train_mask.size()[0])
#del data.train_mask
print(data)

model_init_param={'num_layers': 3, 'hidden_unit': 64, 'dropout_rate': 0.5}

acc_dict=experiment(model_init_param=model_init_param,
                    specify_data=True,data=data,remake_data_mask=False,
                    dataset_split_mode='classification')
print(acc_dict['ACC'])  #输出accuracy值

#示例输出：
#cornell()
#Data(edge_index=[2, 298], test_mask=[183, 10], train_mask=[183, 10], val_mask=[183, 10], x=[183, 1703], y=[183])
#0.7105263157894737（可能每次运行得到的结果都不同）