#通过experiment内置直接通过数据集名称及路径，使用RD2PD类调用图数据
#cora数据集在MLP模型上运行，打印准确率

import sys
sys.path.extend(['whj_code2/integration_experiment'])
from rgb_experiment import experiment,RD2PD

#experiment函数可以自动通过RD2PD类传参，不需要显式调用RD2PD类
model_init_param={'num_layers': 2, 'hidden_unit': 16, 'dropout_rate': 0.5}
model_name='gcn'
dataset_name='bgp'

acc_dict=experiment(model_init_param=model_init_param,dataset_name=dataset_name,
                    dataset_split_mode='ratio',model_name=model_name,
                    dataset_split_seed=14530529,learning_rate=0.01,epoch=500,
                    early_stopping=100,weight_decay=0.0005)
print(acc_dict)  #输出accuracy值


#输出结果示例：0.856353591160221（可能每次运行得到的结果都不同）