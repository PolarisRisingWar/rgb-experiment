#所有可用图数据集在所有模型上以10次以5-2-3为数据集划分比例、以默认超参数作为超参数的准确度平均值

#调整超参
to_un=False  #如果置True则跑所有数据集上原始方向，否则跑有向图转化为无向图
dataset_split_ratio='5-2-3'
seed_number=10
cuda_index=0
learning_rate=0.01
epoch=300
first_sentence='所有模型，所有数据集，有向图，5-2-3, 10次平均'
output_file='whj_code2/integration_experiment/examples/adb_output2.out'

#包导入
import sys
sys.path.extend(['whj_code2/integration_experiment'])
from rgb_experiment import experiment
from rgb_experiment import InitialParameters

import random

import datetime

#参数配置
all_dataset=InitialParameters.dataset_names1
directed_dataset=['Github','Elliptic','Film','Wiki','Weibo','bgp','ssn5','ssn7',
                'Wisconsin','Texas','Cornell','ogbn_arxiv']
model_list=InitialParameters.model_names


starttime = datetime.datetime.now()
print(starttime)
seeds=[random.randint(0,100000000) for i in range(seed_number)]
file_handle=open(output_file,mode='a')  #追加
file_handle.write(first_sentence+':\n')
dn_list=all_dataset if to_un else directed_dataset
file_handle.write('\t')
for m in model_list:
    file_handle.write(m+'\t')
file_handle.write('C&S\t')
file_handle.write('\n')
for dn in range(len(dn_list)):  #遍历数据集
    d=dn_list[dn]
    file_handle.write(d)
    for i in range(len(model_list)):  #遍历模型
        acc_list=[]
        try:
            for seed in seeds:  #遍历seed_number个数据集划分
                acc_dict=experiment(model_init_param=InitialParameters.default_init_params[i],
                                    dataset_split_seed=seed,dataset_name=d,
                                    model_name=model_list[i],
                                    to_undirected_graph=False if to_un else True,
                                    learning_rate=learning_rate,epoch=epoch,
                                    need_all_metrics=False,cuda_index=cuda_index,
                                    print_print=False)
                acc_list.append(acc_dict['ACC'])
            file_handle.write('\t'+str(round(sum(acc_list)/seed_number,3)))
        except RuntimeError:  #一般来说就是OOM了……
            file_handle.write('\tOOM')
    
    #MLP+C&S
    acc_list=[]
    for seed in seeds:  #遍历seed_number个数据集划分
        acc_dict=experiment(model_init_param={'num_layers':3,'hidden_unit':64,'dropout_rate':0.5},
                            dataset_split_seed=seed,dataset_name=d,
                            model_name='mlp',
                            post_cs=True,cs_param=InitialParameters.default_cs_param,
                            to_undirected_graph=False if to_un else True,
                            learning_rate=learning_rate,epoch=epoch,
                            need_all_metrics=False,cuda_index=cuda_index,
                            print_print=False)
        acc_list.append(acc_dict['ACC'])
    file_handle.write('\t'+str(round(sum(acc_list)/seed_number,3)))

    file_handle.write('\n')
        
file_handle.close()

endtime = datetime.datetime.now()
print(endtime)
print((endtime - starttime).seconds)