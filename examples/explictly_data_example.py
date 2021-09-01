#通过PyG.datasets直接获得Data格式数据（也可以通过其他方式获取数据）

import sys

sys.path.extend(['whj_code2/integration_experiment'])

from rgb_experiment import experiment

from torch_geometric.datasets import WebKB

#导入Data数据
ds_path='/data/wanghuijuan/dataset1/pyg_ds'
dataset=WebKB(ds_path,"Cornell")
print(dataset)
data=dataset.data
print(data)
#注意该数据的 train_mask / val_mask / test_mask 格式不符合要求
#因此后文在 experiment() 函数中传入remake_data_mask=True这一参数来重置mask

model_init_param={'num_layers': 3, 'hidden_unit': 64, 'dropout_rate': 0.5}

acc_dict=experiment(model_init_param=model_init_param,
                    specify_data=True,data=data,remake_data_mask=True)
print(acc_dict['ACC'])  #输出accuracy值

#示例输出：
#cornell()
#Data(edge_index=[2, 298], test_mask=[183, 10], train_mask=[183, 10], val_mask=[183, 10], x=[183, 1703], y=[183])
#0.7105263157894737（可能每次运行得到的结果都不同）