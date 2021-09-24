#通过experiment内置直接通过数据集名称及路径，使用RD2PD类调用图数据
#cora数据集在MLP+C&S模型上运行，打印准确率

import sys
sys.path.extend(['whj_code2/integration_experiment'])
from rgb_experiment import experiment

#experiment函数可以自动通过RD2PD类传参，不需要显示调用RD2PD类
model_init_param={'num_layers': 3, 'hidden_unit': 64, 'dropout_rate': 0.5}
model_name='mlp'
dataset_name='cora'

acc_dict=experiment(model_init_param=model_init_param,dataset_name=dataset_name,
                    dataset_split_mode='ratio',model_name=model_name,
                    post_cs=True,
                    cs_param={'num_correction_layers': 50, 'correction_alpha': 0.8,
                    'num_smoothing_layers': 50, 'smoothing_alpha': 0.8, 'autoscale': True})
print(acc_dict['ACC'])  #输出accuracy值

#输出结果示例：0.7097966728280961（可能每次运行得到的结果都不同）