#通过experiment内置直接通过数据集名称及路径，使用zjutoid类调用图数据
#cora数据集在MLP模型上运行，打印准确率

import sys
sys.path.extend(['whj_code2/integration_experiment'])
from rgb_experiment import experiment

#experiment函数可以自动通过zjutoid类传参，不需要显示调用zjutoid类
model_init_param={'num_layers': 3, 'hidden_unit': 64, 'dropout_rate': 0.5}
model_name='MLP'
dataset_name='cora'

acc_dict=experiment(model_init_param=model_init_param,dataset_name=dataset_name,
                    dataset_split_mode='random',check_data_valid=False)
print(acc_dict['ACC'])  #输出accuracy值

#输出结果示例：0.7097966728280961（可能每次运行得到的结果都不同）