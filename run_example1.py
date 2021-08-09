#在initial_params.py中加载的所有数据集上运行baseline

from itexperiments import experiment
from initial_params import InitialParameters

import random

cuda_index=0

learning_rate=0.01
epoch=300

#随机生成seed_number个整数作为种子，跑模型：
seed_number=10
#seeds=[88444839, 57044214, 61447882, 85447514, 49198713, 74564747, 16901785, 9209019, 82043533, 13271186]
seeds=[random.randint(0,100000000) for i in range(seed_number)]

#"""
file_handle=open('whj_code2/integration_experiment/run_example1_output.out',
                mode='a')  #追加
file_handle.write('新数据集有向图Weibo，lr0.01+epoch300+10，无向图版本:\n')
#"""

for dn in [12]:  #遍历数据集
    d=InitialParameters.dataset_name_root_map[dn]
    file_handle.write(d['dataset_name'])
    for i in range(6):  #遍历模型
        #print(d)
        #print(i)
        #上述两个主要可用于监测可能发生的在某一步报bug（比如GPU被挤出去了（help me））
        acc_list=[]
        try:
            for seed in seeds:  #遍历seed_number个数据集划分
                acc_dict=experiment(model_init_param=InitialParameters.default_init_params[i],
                                    dataset_split_seed=seed,
                                    model_name=InitialParameters.model_names[i],
                                    to_undirected_graph=True,
                                    learning_rate=learning_rate,epoch=epoch,
                                    pics_root='whj_code2/integration_experiment/pics/di',
                                    print_pics=False,pics_name='di_'+d['dataset_name']+InitialParameters.model_names[i]+'.png',
                                    vis_feat=False,feat_pic_names_prefix='di_'+d['dataset_name']+InitialParameters.model_names[i],
                                    cuda_index=cuda_index,**d)
                acc_list.append(acc_dict['ACC'])
            file_handle.write('\t'+str(round(sum(acc_list)/seed_number,3)))
        except RuntimeError:  #一般来说就是OOM了……
            file_handle.write('\tOOM')
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
                            cuda_index=cuda_index,**d)
        acc_list.append(acc_dict['ACC'])
    file_handle.write('\t'+str(round(sum(acc_list)/seed_number,3)))
    #"""

    file_handle.write('\n')
        
file_handle.close()