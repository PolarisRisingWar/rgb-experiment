from matplotlib.font_manager import FontProperties
from zjutoid import zjutoid

from visualize_feature import visualize_feature
from initial_params import InitialParameters

from models import MLP,GCN,GraphSAGE,GAT,GGNN,APPNPStack

import torch
import torch.nn as nn
from torch.functional import Tensor

from torch_geometric.nn import CorrectAndSmooth
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data

from sklearn import metrics

import matplotlib.pyplot as plt

def experiment(model_init_param:dict,
                task:str='node_prediction',
                dataset_name:str='Github',
                dataset_root:str=InitialParameters.default_data_path,
                dataset_split_mode:str='ratio',
                dataset_split_ratio:str='6-2-2',
                dataset_split_seed:int=10,
                cuda_index:int=0,
                train_mode:str='fixed_args',
                criterion:type=nn.NLLLoss,
                model_name:str='MLP',
                model_forward_param:dict=None,
                learning_rate:int=0.1,
                epoch:int=50,
                early_stopping:int=10,
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
                feat_pic_names_prefix:str=None):
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

    返回值：
    {'ACC':accuracy,'precision_score':precision_score,'recall_score':recall_score,
    'f1_score':f1_score}
    TODO:以及调optimizer的超参
    
    TODO：……调一下参数顺序
    TODO:这么多参数是为了先写出来。等后期可以考虑使用**kwargs这种代指
    TODO:提高容错
    TODO:继续解耦
    TODO：交叉多种数据集划分方式以提升结果鲁棒性，这个过程我现在还是放外面的，得想想要不要放里面
    
    """
    if not specify_data:
        dataset=zjutoid(root=dataset_root,name=dataset_name,
                split=dataset_split_mode,ratio=dataset_split_ratio,seed=dataset_split_seed)
        data=dataset.data
        #TODO:dataset会自动输出一些信息。等后期可以考虑优化这部分输出
    else:
        #TODO：检查data中是否有mask参数，如果没有的话手动添加
        #TODO: if remake_data_mask=True,直接重置data的mask
        pass

    if to_undirected_graph:
        data.edge_index = to_undirected(data.edge_index,num_nodes=data.num_nodes)

    device=torch.device('cuda:'+str(cuda_index) if torch.cuda.is_available() else "cpu")
    
    data=data.to(device)

    input_dim=data.num_node_features
    output_dim=data.y.unique().size()[0]

    model_name=model_name.lower()
    if model_name=='mlp':
        model=MLP(input_dim=input_dim,output_dim=output_dim,**model_init_param)
        model_forward_param={'x':data.x}
    elif model_name=='gcn':
        model=GCN(input_dim=input_dim,output_dim=output_dim,**model_init_param)
        model_forward_param={'x':data.x,'edge_index':data.edge_index}
    elif model_name=='graphsage':
        model=GraphSAGE(input_dim=input_dim,output_dim=output_dim,**model_init_param)
        model_forward_param={'x':data.x,'edge_index':data.edge_index}
    elif model_name=='gat':
        model=GAT(input_dim=input_dim,output_dim=output_dim,**model_init_param)
        model_forward_param={'x':data.x,'edge_index':data.edge_index}
    elif model_name=='ggnn':
        model=GGNN(input_dim=input_dim,output_dim=output_dim,**model_init_param)
        model_forward_param={'x':data.x,'edge_index':data.edge_index}
    elif model_name=='appnpstack':
        model=APPNPStack(input_dim=input_dim,output_dim=output_dim,**model_init_param)
        model_forward_param={'x':data.x,'edge_index':data.edge_index}
    
    model.to(device)
    optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)
    #TODO:optimizer也作为可选超参
    criterion=criterion()
    y=data.y
    train_mask=data.train_mask
    val_mask=data.val_mask
    test_mask=data.test_mask
    early_stopping_count=0
    before_lowest_val_loss=0
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
        out=model(**model_forward_param)['out']
        loss=criterion(out[train_mask],y[train_mask])

        train_losses.append(loss.item())
        train_accs.append(compare_pred_label(out[train_mask].max(dim=1)[1],y[train_mask])['ACC'])
        
        loss.backward()
        optimizer.step()

        test_dict=test(model,model_forward_param,y,val_mask)
        val_acc=test_dict['ACC']
        val_accs.append(val_acc)
        val_loss=criterion(test_dict['test_op'][val_mask],y[val_mask])
        val_losses.append(val_loss.item())

        test_dict=test(model,model_forward_param,y,test_mask)
        test_accs.append(test_dict['ACC'])
        test_loss=criterion(test_dict['test_op'][test_mask],y[test_mask])
        test_losses.append(test_loss.item())

        #早停
        if i==0:
            before_lowest_val_loss=val_loss

        if val_loss<=before_lowest_val_loss:
            early_stopping_count=0
            before_lowest_val_loss=val_loss
            best_model=model.state_dict()
        else:
            early_stopping_count+=1
            if early_stopping_count>early_stopping:
                break
    
    model.load_state_dict(best_model)
    metric_result=test(model,model_forward_param,y,data.test_mask)

    if not post_cs:
        #print('ACC:'+str(round(test_acc['ACC'],3)))
        pass
    else:
        #print('原predictor上的ACC：'+str(round(test_acc['ACC'],3)))
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

        metric_result=compare_pred_label(pred,label)
    
    if print_pics:
        font = FontProperties(fname=InitialParameters.simhei_ttf_path)
        plt.rcParams['axes.unicode_minus'] = False

        #TODO:plt.title内容也可以作为入参传入
        plt.title(dataset_name+'数据集在'+model_name+'模型上的ACC和loss',fontproperties=font)

        plt.plot(train_losses, label="training loss")
        plt.plot(train_accs, label="training acc")
        plt.plot(val_losses, label="validating loss")
        plt.plot(val_accs, label="validating acc")
        plt.plot(test_losses, label="testing loss")
        plt.plot(test_accs, label="testing acc")
        plt.legend()

        plt.savefig(pics_root+'/'+pics_name)

        plt.close()
        #如果不加这句的话，如果print_pics和vis_feat同时置True，可视化特征的图上就会出现这边的图
    
    if print_confusion_matrix:
        #打印测试集上的混淆矩阵
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
    



def test(model,x,y,mask):
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
    
    metric_result=compare_pred_label(pred,label)

    return {'ACC':metric_result['ACC'],'test_op':out,
    'precision_score':metric_result['precision_score'],
    'recall_score':metric_result['recall_score'],'f1_score':metric_result['f1_score'],
    'pred':pred,'label':label,'emb':pure_out['emb']}



def compare_pred_label(pred,label):
    #accuracy
    total = pred.size()[0]
    #print(pred.size())
    correct = pred.eq(label).sum().item()
    accuracy=correct/total

    #判断pred和label是否是tensor，如果是的话需要挪到CPU上
    if isinstance(pred,Tensor):
        p=pred.cpu()
    if isinstance(label,Tensor):
        l=label.cpu()
    
    precision_score=metrics.precision_score(l, p, average='macro',zero_division=0)
    recall_score=metrics.recall_score(l, p, average='macro',zero_division=0)
    f1_score=metrics.f1_score(l, p, average='macro',zero_division=0)

    return {'ACC':accuracy,'precision_score':precision_score,
    'recall_score':recall_score,'f1_score':f1_score}