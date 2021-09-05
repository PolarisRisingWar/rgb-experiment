import sys
sys.path.extend(['whj_code2/integration_experiment'])

from rgb_experiment import RD2PD
from rgb_experiment import experiment

z=RD2PD('Dblp','/data/wanghuijuan/dataset2/rd2pd_ds',specify_non_label_mask=False,
        apply_sample=False,remove_non_label_node=False,split_seed=123456789)
print(z)
print(z.data)

model_init_param={'num_layers': 2, 'hidden_unit': 64, 'dropout_rate': 0.5}
acc_dict=experiment(model_init_param=model_init_param,model_name='gcn',
                    specify_data=True,data=z.data,learning_rate=0.01,epoch=300)
print(acc_dict)