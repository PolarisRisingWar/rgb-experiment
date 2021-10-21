from matplotlib.font_manager import FontProperties

import numpy as np

import scipy.sparse as sp
from scipy.sparse import coo_matrix
from torch.nn import Module

from .visualize_feature import visualize_feature
from .initial_params import InitialParameters
from .rd2pd import RD2PD
from .models import MLP,GCN,GraphSAGE,GAT,GGNN,APPNPStack,GraphSAGE2,PTA,DAGNN,SuperGAT,SGC, \
                    GIN,FAGCN
from .utils import get_whole_mask,get_classification_mask,get_random_mask

import torch
import torch.nn as nn
from torch.functional import Tensor
import torch.nn.functional as F

from torch_geometric.nn import CorrectAndSmooth
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data

from sklearn import metrics

import matplotlib.pyplot as plt

from copy import copy, deepcopy

import random

class TransferMessage():
    """用来在整个py文件中传递变量"""
    def __init__(self) -> None:
        self.need_all_metrics=False
        self.f1_average='macro'
        self.data_device='cpu'

tm=TransferMessage()

def experiment(model_init_param:dict,*,
                task:str='node_prediction',
                dataset_name:str='Github',
                dataset_root:str=InitialParameters.default_data_path,
                dataset_split_mode:str='ratio',
                dataset_split_ratio:str='6-2-2',
                num_train_per_class:int=20,num_val:int=500,num_test:int=1000,
                dataset_split_seed:int=123456789,
                specify_data:bool=False,data:Data=None,remake_data_mask:bool=False,
                to_undirected_graph:bool=False,
                normalize_feature:str=None,normalize_feature_method:str=None,
                pta_loss_decay:float=0.05,
                pta_weight_decay:float=0.005,
                supergat_graph_lambda:float=4,
                cuda_index:int=0,use_cpu:bool=False,
                need_to_reappear:bool=False,reappear_seed:int=14530529,
                model_name:str='MLP',
                learning_rate:float=0.1,weight_decay:float=0,
                epoch:int=50,
                early_stopping:int=10,
                early_stopping_criterion:str='acc',
                implement_early_stopping:bool=True,
                post_cs:bool=False,
                cs_param:dict=None,
                print_pics:bool=False,
                pics_root:str=InitialParameters.default_pics_path,
                pics_name:str='pic1.png',
                print_confusion_matrix:bool=False,
                check_data_valid:bool=False,
                vis_feat:bool=False,
                feat_pic_names_prefix:str=None,
                total_seed:int=12345678,
                ini_seed:int=1234567,
                need_all_metrics:bool=True,
                f1_average:str='macro',
                loss_func_hp:dict=None,print_print:bool=True,
                specify_model:bool=True,model:Module=None,
                begin_early_stopping:int=20):
    """
    入参：
    必写：
    model_init_param：传入模型__init__()的必要超参

    可选：
    任务类型：
    task: node-prediction / link-prediction / graph-classficiation etc.
        TODO：像CogDL的SUPPORTED_DATASETS和SUPPORTED_DATASETS一样
        其实只能实现node classification……
    
    数据集：
    数据集导入：
    隐式调用RD2PD类导入数据集（具体可参考integration_experiment/rgb_experiment/rd2pd.py介绍）：
    dataset_name: github / cora etc.
    dataset_root: /data/wanghuijuan/dataset2/rd2pd_ds etc.
        要求该目录下放置名为dataset_name的文件夹，该文件夹中放置x.npy等原始数据文件
    dataset_split_mode: 
        ratio / classification
            dataset_split_ratio: 数据集划分比例。不严格要求加起来是10
            dataset_split_seed：数据集划分随机种子
        random
            num_train_per_class：每一类选这么多节点作为训练集
            num_val：验证集节点数
            num_test：测试集节点数
            dataset_split_seed：数据集划分随机种子
    显式调用torch_geometric.data.Data的数据集：
    specify_data: 置True时使用 data 超参导入的Data数据
        如果经检测发现data中没有train_mask等三个mask属性或形制不符要求，或置remake_data_mask=True
        则使用dataset_split_mode的方法进行数据集划分

    数据集预处理：
    to_undirected_graph:如果置True，就将data转为无向图
        （因为PyG的to_undirected函数不影响无向图，所以不用区分原图是不是无向图）
    normalize_feature: None 'row' 'col' 'all'
    normalize_feature_method: None 'MinMaxScalar' 'StandardScalar'

    训练：
    训练设备：
    cuda_index：使用GPU时cuda的编号（int或str格式都可以）
    use_cpu: 如置True则使用CPU而非GPU
    训练可复现性设置：
    need_to_reappear: 如置True则设置随机种子reappear_seed使模型具有相比不设置更高的可复现性
    模型：
    model_name: 模型名称（大小写不限）
    model_init_param:model.__init__()中传入的超参
    训练过程：
    pre-processing:
    TODO：检测模型有没有内置的post_ps函数
    training:
    post-processing:
    TODO：把PTA和C&S的都整合到这个里面来（就都用PTA的格式，但像C&S一样简洁。最好能像
        训练的模型一样整合到一起）
    post_cs:要不要在已有的predictor基础上加C&S (correct and smooth)模型
    cs_param: C&S的参数
    TODO: 在post-ps中放置C&S模型，以及拆分correct和smooth两个函数
    训练过程：
    TODO: 损失函数：
    默认使用模型的loss_function函数，如无则使用nn.NLLLoss()（默认值）
        loss_function函数中的超参通过loss_func_hp入参传入
    早停：
        implement_early_stopping: 如置True则应用早停功能
        early_stopping: 早停超过标准这么多epoch后就停止训练
        early_stopping_criterion: 早停标准
    
    训练完成模型后的打印输出工作：
    print_pics：是否要打印loss、accs等的图像
    print_confusion_matrix：是否要打印混淆矩阵
    check_data_valid：是否要检验data的mask三种数据集的数量，以及各自之间没有交集的情况
    vis_feat：是否要将节点特征可视化（可视化节点初始特征，和经卷积后的节点嵌入）
        feat_pic_names_prefix节点特征可视化输出图的名称（前缀，后面加123等）
    need_all_metrics: 如果置False，则只计算ACC的值，其他指标都置0
    
    

    返回值：
    {'ACC':accuracy,'precision_score':precision_score,'recall_score':recall_score,
    'f1_score':f1_score}
    TODO:以及调optimizer的超参
    
    TODO：……调一下参数顺序
    TODO:提高容错
    TODO:继续解耦
    TODO:我想了一下，感觉像device和need_all_metrics这种参数比较适合作为全局变量
    下次想个办法搞个class，就能用self的属性了
    
    """
    if print_print:
        print('正在运行'+(dataset_name if not specify_data else '自定义')+'数据集在'+model_name+'上的节点分类任务模型...')



    #传递参数值
    tm.need_all_metrics=need_all_metrics
    tm.f1_average=f1_average



    #导入数据集
    if not specify_data:
        dataset=RD2PD(dataset_root=dataset_root,dataset_name=dataset_name,
                split_method=dataset_split_mode,split_ratio=dataset_split_ratio,
                split_seed=dataset_split_seed,num_train_per_class=num_train_per_class,
                num_val=num_val,num_test=num_test)
        data=dataset.data
        #print(data)
        #print(data.x[10])
    else:
        data=data.clone()  #这个是为了防止后续对data的处理工作影响原数据
        assert type(data)==Data  #理论上说不是也行但是我没考虑这种情况

        #如果mask不存在或
        #入参remake_data_mask为True
        #或mask虽然存在但形制不符（只接受节点索引的list或者Tensor，或者mask的Tensor
        #list要求长度不大于node_num，Tensor要求维度为1且其长度不大于node_num）
        #则重新配置mask
        mask_exist=hasattr(data,'train_mask') and hasattr(data,'val_mask') \
                    and hasattr(data,'test_mask')
        if mask_exist:
            node_num=data.num_nodes
            islist=isinstance(data.train_mask,list) and isinstance(data.val_mask,list) and \
                    isinstance(data.test_mask,list)
            valid_list=islist and len(data.train_mask)<=node_num and \
                len(data.val_mask)<=node_num and len(data.test_mask)<=node_num
            istensor=isinstance(data.train_mask,Tensor) and isinstance(data.val_mask,Tensor) \
                    and isinstance(data.test_mask,Tensor)
            valid_tensor=istensor and len(data.train_mask.size())==1 and \
                        data.train_mask.size()[0]<=node_num and len(data.val_mask.size())==1 \
                        and data.val_mask.size()[0]<=node_num and \
                        len(data.test_mask.size())==1 and data.test_mask.size()[0]<=node_num
        if remake_data_mask or (not mask_exist) or (not valid_list) or (not valid_tensor):
            #直接重置data的mask
            if print_print:
                print('重新进行数据集划分，配置mask属性ing...')
            if dataset_split_mode=='ratio':
                (train_mask,val_mask,test_mask)=get_whole_mask(data.y,dataset_split_ratio,
                                                dataset_split_seed)
            elif dataset_split_mode=='classification':
                (train_mask,val_mask,test_mask)=get_classification_mask(data.y,
                                                dataset_split_ratio,dataset_split_seed)
            elif dataset_split_mode=='random':
                (train_mask,val_mask,test_mask)=get_random_mask(data.y,num_train_per_class,
                                                num_val,num_test,dataset_split_seed)
            data.train_mask=train_mask
            data.val_mask=val_mask
            data.test_mask=test_mask
            if print_print:
                print('mask配置成功！')
    if print_print:
        print('数据集导入成功！')
    


    #数据集预处理
    #有向图转换为无向图
    if to_undirected_graph:
        if print_print:
            print('正在将图数据转换为无向图...')
        data.edge_index = to_undirected(data.edge_index,num_nodes=data.num_nodes)
        if print_print:
            print('转换成功！')


    
    #训练
    #训练设备
    device=torch.device('cuda:'+str(cuda_index) if (torch.cuda.is_available() and not use_cpu) \
                        else "cpu")
    


    #传递参数值
    tm.data_device=device


    
    #数据集预处理
    #特征归一化
    data=data.to(device)
    features=data.x
       
    if normalize_feature in ['row','col','all']:
        norm_feat_dim_map={'row':1,'col':0,'all':[0,1]}
        norm_feat_dim=norm_feat_dim_map[normalize_feature]
        if print_print:
            print('正在进行特征在'+normalize_feature+'维度上的'+normalize_feature+'归一化...')
        if normalize_feature_method=='MinMaxScalar':
            
            if isinstance(norm_feat_dim,int):
                feat_min=torch.min(features,norm_feat_dim)[0]
                feat_max=torch.max(features,norm_feat_dim)[0]
            else:
                feat_min=torch.min(features)
                feat_max=torch.max(features)
            
            denominator=(feat_max-feat_min).clamp(min=1e-12)
            #min值参考了：https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html

            if norm_feat_dim==1:
                features=(features.T-feat_min)/denominator
                features=features.T
            else:
                features=(features-feat_min)/denominator
            
        elif normalize_feature_method=='StandardScalar':
            feat_mean=torch.mean(features,norm_feat_dim)
            feat_std=torch.std(features,norm_feat_dim)
            denominator=feat_std.clamp(min=1e-12)
            if norm_feat_dim==1:
                features=(features.T-feat_mean)/denominator
                features=features.T
            else:
                features=(features-feat_mean)/denominator

        else:
            #以下代码参考了：https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/normalize_features.html
            features=features/features.sum(norm_feat_dim, keepdim=True).clamp(min=1)
            #顺便给出PyTorch的实现以供参考：https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html#torch.nn.functional.normalize
        if print_print:
            print('完成归一化工作！')



    #训练
    #模型可复现性
    if need_to_reappear:
        random.seed(reappear_seed)
        np.random.seed(reappear_seed)
        torch.manual_seed(reappear_seed)
        if torch.cuda.is_available(): 
            torch.cuda.manual_seed(reappear_seed)



    #模型
    input_dim=data.num_node_features
    output_dim=data.y.max().item()+1  #不用之前的unique().size()[0]是考虑到我用-1指无标签了

    

    #pre-processing

    #training

    #post-processing

    model_name=model_name.lower()
    if model_name=='mlp':
        model=MLP(input_dim=input_dim,output_dim=output_dim,**model_init_param)
        model_forward_param={'x':features}
    elif model_name=='gcn':
        model=GCN(input_dim=input_dim,output_dim=output_dim,**model_init_param)
        model_forward_param={'x':features,'edge_index':data.edge_index}
    elif model_name=='graphsage':
        model=GraphSAGE(input_dim=input_dim,output_dim=output_dim,**model_init_param)
        model_forward_param={'x':features,'edge_index':data.edge_index}
    elif model_name=='graphsage2':
        model=GraphSAGE2(input_dim=input_dim,output_dim=output_dim,**model_init_param)
        model_forward_param={'x':features,'edge_index':data.edge_index}
    elif model_name=='gat':
        model=GAT(input_dim=input_dim,output_dim=output_dim,**model_init_param)
        model_forward_param={'x':features,'edge_index':data.edge_index}
    elif model_name=='ggnn':
        model=GGNN(input_dim=input_dim,output_dim=output_dim,**model_init_param)
        model_forward_param={'x':features,'edge_index':data.edge_index}
    elif model_name=='appnpstack':
        model=APPNPStack(input_dim=input_dim,output_dim=output_dim,**model_init_param)
        model_forward_param={'x':features,'edge_index':data.edge_index}
    elif model_name=='dagnn':
        model=DAGNN(input_dim=input_dim,output_dim=output_dim,**model_init_param)
        model_forward_param={'x':features,'edge_index':data.edge_index}
    elif model_name=='pta':
        edge_index=data.edge_index.cpu()

        adj=edge_index2sparse_matrix(edge_index,data.num_nodes)
        adj = adj + sp.eye(adj.shape[0])
        adj = normalize_adj(adj)
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        adj=adj.to(device)

        idx_train=data.train_mask.nonzero(as_tuple=True)[0]
        idx_val=data.val_mask.nonzero(as_tuple=True)[0]
        idx_test=data.test_mask.nonzero(as_tuple=True)[0]

        labels=data.y

        y_soft_train = label_propagation(adj, labels, idx_train, model_init_param['K'], 
                        model_init_param['alpha'],device)
        y_soft_val = label_propagation(adj, labels, idx_val, model_init_param['K'], 
                        model_init_param['alpha'],device)
        y_soft_test = label_propagation(adj, labels, idx_test, model_init_param['K'], 
                        model_init_param['alpha'],device)

        model=PTA(nfeat=input_dim,nclass=output_dim,**model_init_param)
    elif model_name=='supergat':
        model=SuperGAT(input_dim=input_dim,output_dim=output_dim,**model_init_param)
        model_forward_param={'x':features,'edge_index':data.edge_index}
    elif model_name=='sgc':
        model=SGC(input_dim=input_dim,output_dim=output_dim,**model_init_param)
        model_forward_param={'x':features,'edge_index':data.edge_index}
    elif model_name=='gin':
        model=GIN(input_dim=input_dim,output_dim=output_dim,**model_init_param)
        model_forward_param={'x':features,'edge_index':data.edge_index}
    elif model_name=='fagcn':
        model=FAGCN(input_dim=input_dim,output_dim=output_dim,**model_init_param)
        model_forward_param={'x':features,'edge_index':data.edge_index}

        
    
    model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    #TODO:optimizer也作为可选超参

    if hasattr(model,'loss_function'):
        if print_print:
            print('model含有loss_function函数！')
    else:
        if print_print:
            print('model中不含loss_function函数，使用NLLLoss作为损失函数')
    criterion=nn.NLLLoss()
    y=data.y
    train_mask=data.train_mask
    val_mask=data.val_mask
    test_mask=data.test_mask
    early_stopping_count=0
    before_lowest_val_loss=0
    before_highest_val_acc=0
    #TODO:增加用acc+loss来进行早停的功能（类似APPNP的实验设置）
    val_accs=[]
    val_losses=[]
    best_model={}
    train_accs=[]
    train_losses=[]
    test_accs=[]
    test_losses=[]

    for i in range(epoch):
        
        model.train()
        optimizer.zero_grad()
        
        if model_name=='pta':
            output = model(features)
            loss = pta_loss_decay * model.loss_function(y_hat = output, y_soft = y_soft_train, 
                    epoch = i) + pta_weight_decay * torch.sum(model.Linear1.weight ** 2) / 2
        else:
            model_out=model(**model_forward_param)
            out=model_out['out']
            loss=criterion(out[train_mask],y[train_mask])

            if model_name=='supergat':
                loss += supergat_graph_lambda * model_out['att_loss']

            train_accs.append(compare_pred_label(out[train_mask].max(dim=1)[1],y[train_mask],
                            need_all_metrics)['ACC'])

        train_losses.append(loss.item())
        
        loss.backward()
        optimizer.step()

        if model_name=='pta':
            output = model.inference(output, adj)
            train_accs.append(compare_pred_label(output[idx_train].max(dim=1)[1],y[idx_train],
                            need_all_metrics)['ACC'])

            model.eval()
            output = model(features)

            val_loss = pta_loss_decay * model.loss_function(y_hat = output, y_soft = y_soft_val)
            val_losses.append(val_loss.item())
            output = model.inference(output, adj)
            val_acc=compare_pred_label(output[idx_val].max(dim=1)[1],y[idx_val],
                                        need_all_metrics)['ACC']
            val_accs.append(val_acc)

            loss_test = pta_loss_decay * model.loss_function(y_hat = output, 
                                                            y_soft = y_soft_test)
            test_losses.append(loss_test.item())
            metric_result=compare_pred_label(output[idx_test].max(dim=1)[1],y[idx_test],
                                            need_all_metrics)
            test_accs.append(metric_result['ACC'])
        else:
            val_dict=test(model,model_forward_param,y,val_mask,need_all_metrics)
            val_acc=val_dict['ACC']
            val_accs.append(val_acc)
            val_loss=criterion(val_dict['test_op'][val_mask],y[val_mask])
            val_losses.append(val_loss.item())

            test_dict=test(model,model_forward_param,y,test_mask,need_all_metrics)
            test_accs.append(test_dict['ACC'])
            test_loss=criterion(test_dict['test_op'][test_mask],y[test_mask])
            test_losses.append(test_loss.item())
        
        #if i%10==0:
        #    print('第 '+str(i)+' epoch '+str(test_dict['ACC']))

        #早停
        if i==0:
            before_lowest_val_loss=val_loss
            before_highest_val_acc=val_acc

        if early_stopping_criterion=='loss':
            if val_loss<=before_lowest_val_loss:
                early_stopping_count=0
                before_lowest_val_loss=val_loss
                best_model=deepcopy(model.state_dict())
                if model_name=='pta':
                    best_metric_result=deepcopy(metric_result)
            elif implement_early_stopping and i>begin_early_stopping:
                early_stopping_count+=1
                if early_stopping_count>early_stopping:
                    break
        elif early_stopping_criterion=='acc':
            if val_acc>=before_highest_val_acc:
                early_stopping_count=0
                before_highest_val_acc=val_acc
                best_model=deepcopy(model.state_dict())
                if model_name=='pta':
                    best_metric_result=deepcopy(metric_result)
            elif implement_early_stopping and i>begin_early_stopping:
                early_stopping_count+=1
                if early_stopping_count>early_stopping:
                    break
        
    
    model.load_state_dict(best_model)

    if model_name=='pta':
        metric_result=best_metric_result
    else:
        metric_result=test(model,model_forward_param,y,test_mask,need_all_metrics)

    if not post_cs:
        pass
    else:
        assert not model_name=='pta'
        
        #TODO:打印原predictor上的输出，与经C&S后的结果作对比
        post=CorrectAndSmooth(**cs_param)
        y_soft=metric_result['test_op'].exp()
        #我写的全是log_softmax，所以要加这句话
        #TODO：判断一下分类层是log_softmax还是softmax然后搞这句话（可以参考PyG的C&S实现）
        edge_index=data.edge_index
        y_soft = post.correct(y_soft,y[train_mask],train_mask,edge_index)
        y_soft = post.smooth(y_soft,y[train_mask],train_mask,edge_index)

        y_soft = y_soft.max(dim=1)[1]

        test_mask=data.test_mask
        pred = y_soft[test_mask]
        label = y[test_mask]

        metric_result=compare_pred_label(pred,label,need_all_metrics)
    
    if print_pics:
        font = FontProperties(fname=InitialParameters.tnr_ttf_path)
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.sans-serif'] = ['Noto Serif CJK JP']

        #TODO:plt.title内容也可以作为入参传入
        if specify_data:
            dataset_name=''
        plt.title(dataset_name+' dataset on '+model_name+' model loss',fontproperties=font)
        plt.plot(train_losses, label="training loss")
        plt.plot(val_losses, label="validating loss")
        plt.plot(test_losses, label="testing loss")
        plt.legend(prop=font)
        plt.savefig(pics_root+'/loss_'+pics_name)
        plt.close()
        #如果不加这句的话，如果print_pics和vis_feat同时置True，可视化特征的图上就会出现这边的图

        plt.title(dataset_name+' dataset on '+model_name+' model ACC ',fontproperties=font)
        plt.plot(train_accs, label="training acc")
        plt.plot(val_accs, label="validating acc")
        plt.plot(test_accs, label="testing acc")
        plt.legend(prop=font)
        plt.savefig(pics_root+'/acc_'+pics_name)
        plt.close()
    
    if print_confusion_matrix:
        #打印测试集上的混淆矩阵
        assert not model_name=='pta'
        #TODO:新增对PTA模型的支持
        
        if post_cs:
            p=pred
            l=label
        else:
            p=metric_result['pred']
            l=metric_result['label']
        cm=metrics.confusion_matrix(l.cpu(), p.cpu())
        print(cm)
    
    if check_data_valid:
        print('训练集共'+str(data.train_mask.sum().item())+'个数据')
        print('验证集共'+str(data.val_mask.sum().item())+'个数据')
        print('测试集共'+str(data.test_mask.sum().item())+'个数据')
        print('训练集与验证集上有重复的数据共'+str(sum(data.train_mask & data.val_mask).item())+ \
            '个')
        print('训练集与测试集上有重复的数据共'+str(sum(data.train_mask & data.test_mask).item()) \
            +'个')
        print('验证集与测试集上有重复的数据共'+str(sum(data.val_mask & data.test_mask).item())+ \
            '个')
    
    if vis_feat:
        assert not model_name=='pta'
        
        #有没有传入图片名，如果有的话就用，如果没有的话就自定义
        if not isinstance(feat_pic_names_prefix,str):
            feat_pic_names_prefix=dataset_name+'_dataset_'+model_name+'_model'
        
        #输出初始特征
        visualize_feature(data.x,data.y,pics_root,feat_pic_names_prefix+'_initial_feature.png',
                        dataset_name+' dataset initial feature')

        #输出全部数据在best_model上的嵌入向量
        visualize_feature(metric_result['emb'],data.y,pics_root,
                        feat_pic_names_prefix+'_embedding.png',
                        dataset_name+' dataset '+model_name+ \
                        ' model final embedding '+str(round(metric_result['ACC'],3))+'')

    return {'ACC':metric_result['ACC'],'precision_score':metric_result['precision_score'],
    'recall_score':metric_result['recall_score'],'f1_macro':metric_result['f1_macro'],
    'f1_micro':metric_result['f1_micro']}
    #TODO:支持对各种指标计算形式的支持，把这个格式改得更优雅些（跟底下两个函数再次解耦）
    



def test(model,x,y,mask,need_all_metrics):
    """
    model
    x（所有需要传入model.forward函数中的参数，字典形式）
    y
    mask
    返回模型直接输出和评估指标（是按照多标签分类在写的。TODO：二分类要不要多给点？）
    """
    model.eval()
    with torch.no_grad():
        pure_out=model(**x)
        out=pure_out['out']
    label=y
    pred=out.max(dim=1)[1]
    pred=pred[mask]
    label=label[mask]
    
    metric_result=compare_pred_label(pred,label,need_all_metrics)

    return {'ACC':metric_result['ACC'],'test_op':out,
    'precision_score':metric_result['precision_score'],
    'recall_score':metric_result['recall_score'],'f1_macro':metric_result['f1_macro'],
    'f1_micro':metric_result['f1_micro'],
    'pred':pred,'label':label,'emb':pure_out['emb'],'pure_out':pure_out}
    



def compare_pred_label(pred,label,need_all_metrics):
    #accuracy
    total = pred.size()[0]
    #print(pred.size())
    correct = pred.eq(label).sum().item()
    accuracy=correct/total

    if need_all_metrics:
        #判断pred和label是否是tensor，如果是的话需要挪到CPU上
        if isinstance(pred,Tensor):
            p=pred.cpu()
        if isinstance(label,Tensor):
            l=label.cpu()
        
        precision_score=metrics.precision_score(l, p, average='macro',zero_division=0)
        recall_score=metrics.recall_score(l, p, average='macro',zero_division=0)
        #print(tm.f1_average)
        f1_macro=metrics.f1_score(l, p, average='macro',zero_division=0)
        f1_micro=metrics.f1_score(l, p, average='micro',zero_division=0)
    else:
        precision_score=0
        recall_score=0
        f1_score=0

    return {'ACC':accuracy,'precision_score':precision_score,
    'recall_score':recall_score,'f1_macro':f1_macro,'f1_micro':f1_micro}






def edge_index2sparse_matrix(edge_index,node_num):
    """用于PTA模型中将edge_index转换为scipy的coo_matrix格式"""
    sizes=(node_num,node_num)
    v=np.ones(edge_index[0].numel())  #边数
    return coo_matrix((v, edge_index), shape=sizes)

def normalize_adj(mx):
    """用于PTA模型，D^{-\frac{1}{2}}AD^{-\frac{1}{2}}"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """用于PTA模型
    Convert a scipy sparse matrix to a torch sparse tensor.
    顺带一提这个理论上讲是不用这么麻烦的因为我本来就是coo_matrix但是我懒得改了
    TODO：写得简洁一点，毕竟我的稀疏矩阵本来就是coo_matrix啊"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def label_propagation(adj, labels, idx, K, alpha,device):
    """用于PTA模型的标签传播部分"""
    y0 = torch.zeros(size=(labels.shape[0], labels.max().item() + 1)).to(device)
    for i in idx:
        y0[i][labels[i]] = 1.0
    
    y = y0
    
    #对负数标签（即无标签的数据），随便给个标签应付一下，反正不在mask里，没它们的事
    #否则负数标签会导致F.one_hot报错
    y_nonlabel_mask=labels<0
    label_copy=copy(labels)
    label_copy[y_nonlabel_mask]=0

    onehotlabel=F.one_hot(label_copy)

    for _ in range(K): 
        y = torch.matmul(adj, y)  #应该是因为这一步，所以y不会再转回来影响y0了
        for i in idx:
            y[i] = onehotlabel[i]
        y = (1 - alpha) * y + alpha * y0
    return y