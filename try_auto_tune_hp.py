from auto_tune_hp import auto_tune_hp,auto_tune_cs
from initial_params import InitialParameters

#原始参数
model_init_param_mlp={'num_layers':3,'hidden_unit':64,'dropout_rate':0.4}
cs_param={'num_correction_layers':50,'correction_alpha':1,'num_smoothing_layers':50,
        'smoothing_alpha':0.8,'autoscale':True}

hp=[['num_layers',[3,1,2,4,5]],
    ['hidden_unit',[64,50,100]],
    ['dropout_rate',[0.5,0.3,0.8]]]
cs_param=[['num_correction_layers',[50,30,70]],
        ['correction_alpha',[1,0.8,0.9]],
        ['num_smoothing_layers',[50,30,70]],
        ['smoothing_alpha',[0.8,0.9,1]],
        ['scale',[20.,30.,40.]],
        ['autoscale',[False,True]]]

#auto_tune_hp(InitialParameters.dataset_name_root_map[0]['dataset_name'],
#            InitialParameters.dataset_name_root_map[0]['dataset_root'],hp,'mlp')
auto_tune_cs(InitialParameters.dataset_name_root_map[0]['dataset_name'],
            InitialParameters.dataset_name_root_map[0]['dataset_root'],
            {'num_layers': 3, 'hidden_unit': 50, 'dropout_rate': 0.8},'MLP',cs_param)

#调优后的参数：
"""
{'num_layers': 3, 'hidden_unit': 50, 'dropout_rate': 0.8}
{'num_correction_layers': 70, 'correction_alpha': 1, 'num_smoothing_layers': 30, 'smoothing_alpha': 0.9,'autoscale': True}
"""