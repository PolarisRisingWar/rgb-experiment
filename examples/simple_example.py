#通过experiment内置直接通过数据集名称及路径，使用RD2PD类调用图数据
#cora数据集在MLP模型上运行，打印准确率

import sys
sys.path.extend(['whj_code2/integration_experiment'])
from rgb_experiment import experiment

#experiment函数可以自动通过RD2PD类传参，不需要显式调用RD2PD类
model_init_param={'hidden_dim':8,'heads':8,'dropout_rate':0.6,'edge_sample_ratio':0.8,'neg_sample_ratio':0.5}
model_name='supergat'
dataset_name='cora'

acc_dict=experiment(model_init_param=model_init_param,dataset_name=dataset_name,
                    dataset_split_mode='ratio',model_name=model_name,
                    learning_rate=0.005,epoch=500)
print(acc_dict['ACC'])  #输出accuracy值

#输出结果示例：0.856353591160221（可能每次运行得到的结果都不同）