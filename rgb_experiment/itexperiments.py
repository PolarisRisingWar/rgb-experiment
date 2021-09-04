from matplotlib.font_manager import FontProperties

import numpy as np

import scipy.sparse as sp
from scipy.sparse import coo_matrix

from .visualize_feature import visualize_feature
from .initial_params import InitialParameters
from .zjutoid import zjutoid
from .models import MLP,GCN,GraphSAGE,GAT,GGNN,APPNPStack,GraphSAGE2,PTA,DAGNN

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
    def __init__(self) -> None:
        self.f1_average='macro'

tm=TransferMessage()

def experiment(model_init_param:dict,
                pta_loss_decay:float=0.05,
                pta_weight_decay:float=0.005,
                task:str='node_prediction',
                dataset_name:str='Github',
                dataset_root:str=InitialParameters.default_data_path,
                dataset_split_mode:str='ratio',
                dataset_split_ratio:str='6-2-2',
                dataset_split_seed:int=123456789,
                cuda_index:int=0,
                use_cpu:bool=False,
                train_mode:str='fixed_args',
                criterion:type=nn.NLLLoss,
                model_name:str='MLP',
                model_forward_param:dict=None,
                learning_rate:float=0.1,
                epoch:int=50,
                early_stopping:int=10,
                early_stopping_criterion:str='acc',
                implement_early_stopping:bool=True,
                post_cs:bool=False,
                cs_param:dict=None,
                to_undirected_graph:bool=False,
                specify_data:bool=False,
                data:Data=None,
                remake_data_mask:bool=False,
                print_pics:bool=False,
                pics_root:str=InitialParameters.default_pics_path,
                pics_name:str='pic1.png',
                print_confusion_matrix:bool=False,
                check_data_valid:bool=False,
                vis_feat:bool=False,
                feat_pic_names_prefix:str=None,
                normalize_feature:str=None,
                normalize_feature_method:str=None,
                total_seed:int=12345678,
                ini_seed:int=1234567,
                need_to_reappear:bool=False,
                need_all_metrics:bool=True,
                f1_average:str='macro'):
    """
    入参：
    task: node-prediction / link-prediction / graph-classficiation
        TODO：像CogDL的SUPPORTED_DATASETS和SUPPORTED_DATASETS一样
        其实只能实现node classification……
    dataset_name: github / cora etc.
    dataset_root
    dataset_split_mode: public / full / random / ratio 
    dataset_split_ratio:（dataset_split_mode='ratio'）不严格要求加起来是10，三个这种形式的数字就行
    dataset_split_seed:（dataset_split_mode='ratio'）
    cuda_index
    train_mode: 
        fixed_args就跑已设置好的一系列参数
        auto_ml就调用隔壁auto_tune_hp.py来自动调参这样？我再想想
    criterion:损失函数所用的模型（这个感觉要看编码风格，看喜欢在Module里面定义loss还是在外面定义loss）
        PyTorch60分钟速成教程是在外面的，我当年学PyTorch第一套教程就是那个，
        对我PyTorch代码编写的影响力度，就像初恋之对恋爱观的影响力度一样。所以我就写外面了。
        TODO：呀咩咯！配置文件不会要专门写个type格式吧！
    model_name
    model_init_param:model.__init__()中传入的参数
        TODO：默认值咋整？可以通过配置文件输入一套固定的参数？或者模型本来就内置一套固定的默认参数？
        TODO:让我想想这块怎么写比较靠谱。以及参考一下CogDL的实现
    model_forward_param:model.forward()中额外传入的参数（感觉应该不需要，但是先把位置留出来）
    implement_early_stopping：是否应用早停机制
    post_cs:要不要在已有的predictor基础上加correct and smooth
        TODO:感觉把它放在这里怪怪的！！！
            但是我一时之间也不知道不放这里还能放哪里！！！
    cs_param:c&s的参数
    TODO：correct和smooth这两个环节是可拆分的，让我想想怎么拆
    to_undirected_graph:如果置True，就将data转为无向图
        （因为to_undirected不影响无向图，所以不用区分原图是不是无向图）
    specify_data, data:Data=None, remake_data_mask:直接输入一个PyG的Data格式的data
        具体的直接看下面代码即可
    print_pics：是否要打印loss、accs等的图像
    print_confusion_matrix：是否要打印混淆矩阵
    check_data_valid：是否要检验data的mask三种数据集的数量，以及各自之间没有交集的情况
    vis_feat：是否要将节点特征可视化（可视化节点初始特征，和经卷积后的节点嵌入）
        feat_pic_names_prefix节点特征可视化输出图的名称（前缀，后面加123等）
    need_all_metrics: 如果置False，则只计算ACC的值，其他指标都置0
    early_stopping_criterion: acc / loss
    normalize_feature: None 'row' 'col' 'all'
    normalize_feature_method: None 'MinMaxScalar' 'StandardScalar'

    返回值：
    {'ACC':accuracy,'precision_score':precision_score,'recall_score':recall_score,
    'f1_score':f1_score}
    TODO:以及调optimizer的超参
    
    TODO：……调一下参数顺序
    TODO:这么多参数是为了先写出来。等后期可以考虑使用**kwargs这种代指
    TODO:提高容错
    TODO:继续解耦
    TODO：交叉多种数据集划分方式以提升结果鲁棒性，这个过程我现在还是放外面的，得想想要不要放里面
    TODO:我想了一下，感觉像device和need_all_metrics这种参数比较适合作为全局变量
    下次想个办法搞个class，就能用self的属性了
    
    """
    #print(tm.f1_average)
    tm.f1_average=f1_average

    if not specify_data:
        dataset=zjutoid(root=dataset_root,name=dataset_name,
                split=dataset_split_mode,ratio=dataset_split_ratio,seed=dataset_split_seed)
        data=dataset.data
        #TODO:dataset会自动输出一些信息。等后期可以考虑优化这部分输出
    else:
        data=data.clone()  #这个是为了防止影响data原数据（据我测试可以实现这一目标）

        #TODO：检查data中的mask参数，如果没有的话手动添加；如果形制不符（是类似PTA idx那种格式
        #需要修改格式）；如果是其他什么奇奇怪怪的格式直接重置mask
        
        if remake_data_mask:  #直接重置data的mask
            #TODO：把remake_mask函数同步成zjutoid2里的格式
            remake_mask(data,dataset_split_ratio,dataset_split_seed)
        
    if to_undirected_graph:
        #print(data.edge_index)
        #print(data.num_nodes)
        data.edge_index = to_undirected(data.edge_index,num_nodes=data.num_nodes)

    device=torch.device('cuda:'+str(cuda_index) if torch.cuda.is_available() else "cpu")
    if use_cpu:
        device='cpu'
    
    if need_to_reappear:
        random.seed(total_seed)
        np.random.seed(total_seed)
        torch.manual_seed(ini_seed)
        if torch.cuda.is_available(): 
            torch.cuda.manual_seed(ini_seed)

    input_dim=data.num_node_features
    output_dim=data.y.max().item()+1  #不用之前的unique().size()[0]是考虑到我用-1指无标签了
    #print(output_dim)

    data=data.to(device)

    #特征归一化部分
    features=data.x
       
    if normalize_feature in ['row','col','all']:
        #print(features)
        norm_feat_dim_map={'row':1,'col':0,'all':[0,1]}
        norm_feat_dim=norm_feat_dim_map[normalize_feature]

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
            #print(features)
            
        elif normalize_feature_method=='StandardScalar':
            feat_mean=torch.mean(features,norm_feat_dim)
            feat_std=torch.std(features,norm_feat_dim)
            denominator=feat_std.clamp(min=1e-12)
            if norm_feat_dim==1:
                features=(features.T-feat_mean)/denominator
                features=features.T
            else:
                features=(features-feat_mean)/denominator
            #print(features)

        else:
            #以下代码参考了：https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/normalize_features.html
            features=features/features.sum(norm_feat_dim, keepdim=True).clamp(min=1)
            #顺便给出PyTorch的实现以供参考：https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html#torch.nn.functional.normalize

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

        y_soft_train = label_propagation(adj, labels, idx_train, model_init_param['K'], model_init_param['alpha'],device)
        y_soft_val = label_propagation(adj, labels, idx_val, model_init_param['K'], model_init_param['alpha'],device)
        y_soft_test = label_propagation(adj, labels, idx_test, model_init_param['K'], model_init_param['alpha'],device)

        model=PTA(nfeat=input_dim,nclass=output_dim,**model_init_param)

        
    
    model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    #TODO:optimizer也作为可选超参

    criterion=criterion()
    y=data.y
    train_mask=data.train_mask
    val_mask=data.val_mask
    test_mask=data.test_mask
    early_stopping_count=0
    before_lowest_val_loss=0
    before_highest_val_acc=0
    #TODO:增加用acc+loss来进行早停的功能
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
            loss = pta_loss_decay * model.loss_function(y_hat = output, y_soft = y_soft_train, epoch = i) + pta_weight_decay * torch.sum(model.Linear1.weight ** 2) / 2
        else:
            out=model(**model_forward_param)['out']
            loss=criterion(out[train_mask],y[train_mask])
            train_accs.append(compare_pred_label(out[train_mask].max(dim=1)[1],y[train_mask],need_all_metrics)['ACC'])

        train_losses.append(loss.item())
        
        loss.backward()
        optimizer.step()

        if model_name=='pta':
            output = model.inference(output, adj)
            train_accs.append(compare_pred_label(output[idx_train].max(dim=1)[1],y[idx_train],need_all_metrics)['ACC'])

            model.eval()
            output = model(features)

            val_loss = pta_loss_decay * model.loss_function(y_hat = output, y_soft = y_soft_val)
            val_losses.append(val_loss.item())
            output = model.inference(output, adj)
            val_acc=compare_pred_label(output[idx_val].max(dim=1)[1],y[idx_val],need_all_metrics)['ACC']
            val_accs.append(val_acc)

            loss_test = pta_loss_decay * model.loss_function(y_hat = output, y_soft = y_soft_test)
            test_losses.append(loss_test.item())
            metric_result=compare_pred_label(output[idx_test].max(dim=1)[1],y[idx_test],need_all_metrics)
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
            elif implement_early_stopping:
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
            elif implement_early_stopping:
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
        font = FontProperties(fname=InitialParameters.simhei_ttf_path)
        plt.rcParams['axes.unicode_minus'] = False

        #TODO:plt.title内容也可以作为入参传入
        if specify_data:
            dataset_name=''
        plt.title(dataset_name+'数据集在'+model_name+'模型上的loss',fontproperties=font)
        plt.plot(train_losses, label="training loss")
        plt.plot(val_losses, label="validating loss")
        plt.plot(test_losses, label="testing loss")
        plt.legend()
        plt.savefig(pics_root+'/loss_'+pics_name)
        plt.close()
        #如果不加这句的话，如果print_pics和vis_feat同时置True，可视化特征的图上就会出现这边的图

        plt.title(dataset_name+'数据集在'+model_name+'模型上的ACC',fontproperties=font)
        plt.plot(train_accs, label="training acc")
        plt.plot(val_accs, label="validating acc")
        plt.plot(test_accs, label="testing acc")
        plt.legend()
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
        print('训练集与验证集上有重复的数据共'+str(sum(data.train_mask & data.val_mask).item())+'个')
        print('训练集与测试集上有重复的数据共'+str(sum(data.train_mask & data.test_mask).item())+'个')
        print('验证集与测试集上有重复的数据共'+str(sum(data.val_mask & data.test_mask).item())+'个')
    
    if vis_feat:
        assert not model_name=='pta'
        
        #有没有传入图片名，如果有的话就用，如果没有的话就自定义
        if not isinstance(feat_pic_names_prefix,str):
            feat_pic_names_prefix=dataset_name+'_dataset_'+model_name+'_model'
        
        #输出初始特征
        visualize_feature(data.x,data.y,pics_root,feat_pic_names_prefix+'_initial_feature.png',
                        dataset_name+'数据集上的初始特征')

        #输出全部数据在best_model上的嵌入向量
        visualize_feature(metric_result['emb'],data.y,pics_root,
                        feat_pic_names_prefix+'_embedding.png',
                        dataset_name+'数据集在'+model_name+'模型上运行后的特征（ACC为'+str(round(metric_result['ACC'],3))+'）')

    return {'ACC':metric_result['ACC'],'precision_score':metric_result['precision_score'],
    'recall_score':metric_result['recall_score'],'f1_score':metric_result['f1_score']}
    



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
    'recall_score':metric_result['recall_score'],'f1_score':metric_result['f1_score'],
    'pred':pred,'label':label,'emb':pure_out['emb']}



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
        f1_score=metrics.f1_score(l, p, average=tm.f1_average,zero_division=0)
    else:
        precision_score=0
        recall_score=0
        f1_score=0

    return {'ACC':accuracy,'precision_score':precision_score,
    'recall_score':recall_score,'f1_score':f1_score}




def make_mask(ratio,num_nodes,seed):
    train_val_test_list=[int(i) for i in ratio.split('-')]
    random.seed(seed)
    tvt_sum=sum(train_val_test_list)
    tvt_ratio_list=[i/tvt_sum for i in train_val_test_list]
    train_end_index=int(tvt_ratio_list[0]*num_nodes)
    val_end_index=train_end_index+int(tvt_ratio_list[1]*num_nodes)
    
    bs=list(range(num_nodes))
    random.shuffle(bs)
    
    train_mask_index=bs[:train_end_index]
    val_mask_index=bs[train_end_index:val_end_index]
    test_mask_index=bs[val_end_index:]
    
    train_mask=torch.tensor([False for i in range(num_nodes)])
    train_mask[train_mask_index]=True
    val_mask=torch.tensor([False for i in range(num_nodes)])
    val_mask[val_mask_index]=True
    test_mask=torch.tensor([False for i in range(num_nodes)])
    test_mask[test_mask_index]=True

    return (train_mask,val_mask,test_mask)

def check_train_containing(train_mask,y):
    """（仅用于分类任务）检查train_mask中是否含有y中所有的标签"""
    for label in y.unique():
        l=label.item()
        if l not in y[train_mask]:
            return False
    return True

def remake_mask(data,ratio,seed=1234567):
    """直接覆盖原data的train_mask, val_mask, test_mask三个属性"""
    while True:
        (train_mask,val_mask,test_mask)=make_mask(ratio,data.num_nodes,seed)
        if check_train_containing(train_mask,data.y):
            data.train_mask=train_mask
            data.val_mask=val_mask
            data.test_mask=test_mask
            break
        else:
            seed+=1





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