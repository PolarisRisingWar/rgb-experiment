import sys
sys.path.extend(['whj_code2/integration_experiment'])

from rgb_experiment import RD2PD
from rgb_experiment import experiment

z=RD2PD('Github','/data/wanghuijuan/dataset2/rd2pd_ds',split_seed=123456789,split_method='classification')
print(z.data)
print(z.data.y.unique())

model_init_param={'num_layers': 2, 'hidden_unit': 64, 'dropout_rate': 0.5}
acc_dict=experiment(model_init_param=model_init_param,model_name='gcn',
                    specify_data=True,data=z.data,learning_rate=0.01,epoch=300,
                    check_data_valid=False)
print(acc_dict)

#示例输出：
#Data(edge_index=[2, 289003], test_mask=[37700], train_mask=[37700], val_mask=[37700], x=[37700, 4005], y=[37700])
#tensor([0, 1])
#{'ACC': 0.8473879607531158, 'precision_score': 0.8091390403167427, 'recall_score': 0.7764252978027122, 'f1_score': 0.790225581079796}
#（可能每次运行得到的结果都不同）