#在我自己整的数据集上运行baseline

from itexperiments import experiment
from initial_params import InitialParameters
from zjutoid2 import zjutoid2

import traceback

import random

import datetime

starttime = datetime.datetime.now()

cuda_index=2

learning_rate=0.01
epoch=300

#随机生成seed_number个整数作为种子，跑模型：
seed_number=10
#seeds=[88444839, 57044214, 61447882, 85447514, 49198713, 74564747, 16901785, 9209019, 82043533, 13271186]
seeds=[random.randint(0,100000000) for i in range(seed_number)]

#"""
file_handle=open('whj_code2/integration_experiment/run_example2_output2.out',
                mode='a')  #追加
file_handle.write('BGP数据集无向图:\n')
#"""

name_root_map=[('bgp','/data/wanghuijuan/dataset1/zjutoid2_ds')]
for dn in range(len(name_root_map)):  #遍历数据集（1个），其中有向图为[0] bgp
    data=zjutoid2(name_root_map[dn][0],name_root_map[dn][1]).data
    for i in range(8):  #遍历模型（8个）
        #print(i)
        acc_list=[]
        try:
            for seed in seeds:  #遍历seed_number个数据集划分
                acc_dict=experiment(model_init_param=InitialParameters.default_init_params[i],
                                    dataset_split_seed=seed,
                                    model_name=InitialParameters.model_names[i],
                                    specify_data=True,data=data,
                                    to_undirected_graph=True,
                                    learning_rate=learning_rate,epoch=epoch,
                                    need_all_metrics=False,
                                    normalize_feature=None,
                                    cuda_index=cuda_index)
                acc_list.append(acc_dict['ACC'])
            file_handle.write('\t'+str(round(sum(acc_list)/seed_number,3)))
        except RuntimeError as e:  #一般来说就是OOM了……
            file_handle.write('\tOOM')
            #print(str(e))
            #print(traceback.print_exc())
            #pass
    
    #MLP+C&S
    #"""
    acc_list=[]
    for seed in seeds:  #遍历seed_number个数据集划分
        acc_dict=experiment(model_init_param={'num_layers':3,'hidden_unit':64,'dropout_rate':0.5},
                            dataset_split_seed=seed,
                            model_name='MLP',
                            to_undirected_graph=True,
                            learning_rate=learning_rate,epoch=epoch,
                            post_cs=True,cs_param=InitialParameters.default_cs_param,
                            early_stopping_criterion='acc',
                            need_all_metrics=False,
                            normalize_feature=None,
                            cuda_index=cuda_index,
                            specify_data=True,data=data)
        acc_list.append(acc_dict['ACC'])
    file_handle.write('\t'+str(round(sum(acc_list)/seed_number,3)))
    #"""

    file_handle.write('\n')
        
file_handle.close()

endtime = datetime.datetime.now()
print((endtime - starttime).seconds)