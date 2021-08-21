#在我自己整的数据集上运行baseline

from itexperiments import experiment
from initial_params import InitialParameters
from zjutoid2 import zjutoid2

import traceback

import random

import datetime

starttime = datetime.datetime.now()

cuda_index=2
model_num=9

learning_rate=0.01
epoch=300

#随机生成seed_number个整数作为种子，跑模型：
seed_number=10
#seeds=[88444839, 57044214, 61447882, 85447514, 49198713, 74564747, 16901785, 9209019, 82043533, 13271186]
seeds=[random.randint(0,100000000) for i in range(seed_number)]

#"""
file_handle=open('whj_code2/integration_experiment/run_example2_output2.out',
                mode='a')  #追加
file_handle.write('运行一个随机性更强、更正确的版本，BGP数据集有向图转无向图:\n')
#"""

name_root_map=[('bgp','/data/wanghuijuan/dataset1/zjutoid2_ds')]
for dn in range(1):  #遍历数据集（1个），其中有向图为[0] bgp
    accs_list=[]
    oom_model_index=set()
    for seed in seeds:
        data=zjutoid2(name_root_map[dn][0],name_root_map[dn][1],split_ratio='6-2-2',split_seed=seed).data
        acc_list=[]
        for i in range(8):  #遍历模型（8个）
            #print(i)
            try:
                acc_dict=experiment(model_init_param=InitialParameters.default_init_params[i],
                                    model_name=InitialParameters.model_names[i],
                                    specify_data=True,data=data,
                                    to_undirected_graph=True,
                                    learning_rate=learning_rate,epoch=epoch,
                                    need_all_metrics=False,
                                    normalize_feature=None,
                                    cuda_index=cuda_index,f1_average='micro')
                acc_list.append(acc_dict['ACC'])
            except RuntimeError as e:  #一般来说就是OOM了……
                oom_model_index.add(i)
                #print(str(e))
                #print(traceback.print_exc())
                pass
        
        #MLP+C&S
        #"""
        acc_dict=experiment(model_init_param={'num_layers':3,'hidden_unit':64,'dropout_rate':0.5},
                            model_name='MLP',
                            to_undirected_graph=True,
                            learning_rate=learning_rate,epoch=epoch,
                            post_cs=True,cs_param=InitialParameters.default_cs_param,
                            need_all_metrics=False,
                            normalize_feature=None,
                            cuda_index=cuda_index,
                            specify_data=True,data=data,f1_average='micro')
        acc_list.append(acc_dict['ACC'])
        
        accs_list.append(acc_list)
    #"""
    #print(len(accs_list))

    for i in range(model_num):
        if i in oom_model_index:
            file_handle.write('\tOOM')
        else:
            file_handle.write('\t'+str(round(sum([accs_list[s][i] for s in range(seed_number)])/seed_number,3)))

    file_handle.write('\n')
        
file_handle.close()

endtime = datetime.datetime.now()
print((endtime - starttime).seconds)