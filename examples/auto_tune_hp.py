#本函数用于自动为数据集调参（贪心算法，调用experiment函数来运行模型）
import sys
sys.path.extend(['whj_code2/integration_experiment'])
from rgb_experiment import experiment

import random

def auto_tune_hp(dataset_name,dataset_root,hp,model_name):
    """
    调用本函数返回一套基于hp优化的、适用于data在model_name上的超参
    贪心算法，用3次用不同的数据集split来得出一个平均结果，按照hp中的顺序来跑模型。
    如果ACC高出0.001才会更新参数（要不然就，没意义嘛）
    TODO：可以直接输入Data（PyG的Data格式图），然后自己造3次mask
    TODO:其他入参
    hp：理论上说tuple和list应该都可以。每个元素第一个元素是超参的key，第二个是可选的value列表
    注意：C&S要用下面那个函数调
    TODO:可以直接输入初始hp的dict，在此基础上调参
    TODO:如果某个参数超出了允许的范围，输出警告但是可以继续运行其他参数
    """
    seed_number=3  #TODO：把这个作为入参
    
    best_param={}

    #初始化best_param
    for i in range(len(hp)):
        best_param[hp[i][0]]=hp[i][1][0]

    try_param=best_param.copy()

    for k_v in hp:
        k=k_v[0]
        best_acc=0
        for v in k_v[1]:
            try_param[k]=v
            seeds=[random.randint(0,10000) for i in range(seed_number)]
            acc=0
            for seed in seeds:
                acc+=experiment(try_param,dataset_name=dataset_name,dataset_root=dataset_root,
                    model_name=model_name,dataset_split_seed=seed)['ACC']
            acc/=seed_number
            if acc>best_acc+0.001:
                best_acc=acc
                best_param[k]=v
        try_param=best_param.copy()
    print(best_param)
                

def auto_tune_cs(dataset_name,dataset_root,model_param,model_name,cs_param):
    #TODO:autoscale咋手动整？
    seed_number=3  #TODO：把这个作为入参
    
    best_param={}

    #初始化best_param
    for i in range(len(cs_param)):
        best_param[cs_param[i][0]]=cs_param[i][1][0]

    try_param=best_param.copy()

    for k_v in cs_param:
        k=k_v[0]
        best_acc=0
        for v in k_v[1]:
            try_param[k]=v
            seeds=[random.randint(0,10000) for i in range(seed_number)]
            acc=0
            for seed in seeds:
                acc+=experiment(model_param,dataset_name=dataset_name,
                    dataset_root=dataset_root,model_name=model_name,dataset_split_seed=seed,
                    post_cs=True,cs_param=try_param)['ACC']
            acc/=seed_number
            if acc>best_acc+0.001:
                best_acc=acc
                best_param[k]=v
        try_param=best_param.copy()
    print(best_param)