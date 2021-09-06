import sys
sys.path.extend(['whj_code2/integration_experiment'])

from rgb_experiment import RD2PD
from rgb_experiment import experiment

z=RD2PD('Dblp','/data/wanghuijuan/dataset2/rd2pd_ds',split_seed=123456789)
print(z.data)

model_init_param={'num_layers': 2, 'hidden_unit': 64, 'dropout_rate': 0.5}
acc_dict=experiment(model_init_param=model_init_param,model_name='gcn',
                    specify_data=True,data=z.data,learning_rate=0.01,epoch=300)
print(acc_dict)

#示例输出：
#Data(edge_index=[2, 617212], test_mask=[40672], train_mask=[40672], val_mask=[40672], x=[40672, 7202], y=[40672])
#{'ACC': 0.4330669944683467, 'precision_score': 0.4445093857051388, 'recall_score': 0.4240076391530519, 'f1_score': 0.42472214916109274}
#（可能每次运行得到的结果都不同）