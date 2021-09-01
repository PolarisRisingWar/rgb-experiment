#在我自己整的数据集上运行baseline

import sys
sys.path.extend(['whj_code2/integration_experiment'])

from rgb_experiment.itexperiments import experiment
from rgb_experiment.initial_params import InitialParameters
from rgb_experiment.zjutoid2 import zjutoid2
from rgb_experiment.rd2pd import RD2PD

import random

import datetime

starttime = datetime.datetime.now()
print(starttime)

cuda_index=1
model_num=10  #跑了几个模型（最后输出的时候算平均值用）

learning_rate=0.01
epoch=300

#随机生成seed_number个整数作为种子，跑模型：
seed_number=10
#seeds=[88444839, 57044214, 61447882, 85447514, 49198713, 74564747, 16901785, 9209019, 82043533, 13271186]
seeds=[random.randint(0,100000000) for i in range(seed_number)]

#"""
file_handle=open('whj_code2/integration_experiment/run_example2_output2.out',
                mode='a')  #追加
file_handle.write('SSN7有向图转无向图:\n')
#"""

rd2pd_root='/data/wanghuijuan/dataset2/rd2pd_ds'
name_root_map=[('bgp',rd2pd_root),
            ('ssn1',rd2pd_root),
            ('ssn2',rd2pd_root),
            ('ssn3',rd2pd_root),
            ('ssn4',rd2pd_root),
            ('ssn5',rd2pd_root),
            ('ssn6',rd2pd_root),
            ('ssn7',rd2pd_root),]
for dn in [7]:  #遍历数据集（8个）
    file_handle.write(name_root_map[dn][0])
    accs_list=[]
    oom_model_index=set()
    for seed in seeds:
        data=RD2PD(name_root_map[dn][0],name_root_map[dn][1],split_ratio='6-2-2',split_seed=seed).data
        #print(data)
        acc_list=[]
        for i in range(9):  #遍历模型（9个）
            #print(InitialParameters.model_names[i])
            try:
                acc_dict=experiment(model_init_param=InitialParameters.default_init_params[i],
                                    model_name=InitialParameters.model_names[i],
                                    specify_data=True,data=data,
                                    to_undirected_graph=True,
                                    learning_rate=learning_rate,epoch=epoch,
                                    need_all_metrics=False,
                                    normalize_feature=None,
                                    cuda_index=cuda_index,f1_average='micro',
                                    print_confusion_matrix=False)
                #print(data.is_directed())
                #break
                acc_list.append(acc_dict['ACC'])
                #print(acc_dict)
            except RuntimeError as e:  #一般来说就是OOM了……
                oom_model_index.add(i)
                acc_list.append(0)
                print(InitialParameters.model_names[i])
                print(str(e))
                #print(traceback.print_exc())
        
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
                            specify_data=True,data=data,f1_average='micro',
                            print_confusion_matrix=False)
        #print(acc_dict)
        acc_list.append(acc_dict['ACC'])
        #"""
        
        accs_list.append(acc_list)
        #print(seed)
        #print(acc_list)
        #print(accs_list)

    #print(len(accs_list))

    for i in range(model_num):
        if i in oom_model_index:
            file_handle.write('\tOOM')
        else:
            file_handle.write('\t'+str(round(sum([accs_list[s][i] for s in range(seed_number)])/seed_number,3)))

    file_handle.write('\n')
        
file_handle.close()

endtime = datetime.datetime.now()
print(endtime)
print((endtime - starttime).seconds)