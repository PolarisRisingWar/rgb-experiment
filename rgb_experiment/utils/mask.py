#暂时只支持分类任务，且y=-1默认为无标签节点不参与数据划分

import random

import torch
from torch import Tensor



def get_whole_mask(y,ratio:str,seed:int=1234567):
    """对整个数据集按比例进行划分"""
    y_have_label_mask=y!=-1
    total_node_num=len(y)
    y_index_tensor=torch.tensor(list(range(total_node_num)),dtype=int)
    masked_index=y_index_tensor[y_have_label_mask]
    while True:
        (train_mask,val_mask,test_mask)=get_order(ratio,masked_index,total_node_num,seed)
        if check_train_containing(train_mask,y):
            return (train_mask,val_mask,test_mask)
        else:
            seed+=1

def check_train_containing(train_mask,y):
    """（仅用于分类任务）检查train_mask中是否含有y中所有的标签（-1不算）"""
    for label in y.unique():
        l=label.item()
        if l==-1:
            continue
        if l not in y[train_mask]:
            return False
    return True




def get_classification_mask(y,total_node_num,ratio:str,seed:int=1234567):
    """
    对每一类节点，将其对应的节点按比例进行数据集划分，总的mask求与
    返回可以直接赋值到整个data上的mask元组

    入参：
    y=data.y
    total_node_num=data.num_nodes
    """
    train_mask=torch.zeros(total_node_num,dtype=torch.bool)
    val_mask=torch.zeros(total_node_num,dtype=torch.bool)
    test_mask=torch.zeros(total_node_num,dtype=torch.bool)
    count=0
    for label in y.unique():
        l=label.item()
        if l==-1:
            continue
        l_mask=y==l  #对应标签的节点索引为True的一维布尔Tensor
        count+=l_mask.sum()
        l_index=l_mask.nonzero(as_tuple=True)[0]  #对应标签的节点索引的一维Tensor
        (l_train_mask,l_val_mask,l_test_mask)=get_order(ratio,l_index,total_node_num,seed)
        train_mask=train_mask | l_train_mask
        val_mask=val_mask | l_val_mask
        test_mask=test_mask | l_test_mask
    return (train_mask,val_mask,test_mask)



def get_order(ratio:str,masked_index:Tensor,total_node_num:int,seed:int=1234567):
    """
    输入划分比例和原始的索引，输出对应划分的mask元组

    入参：
    ratio格式：'1-1-3'
    masked_index是索引的1维Tensor
    TODO：增加对其他格式masked_index的支持

    返回值：(train_mask,val_mask,test_mask)
    都是长度为总节点数，对应索引置True的布尔Tensor
    """
    random.seed(seed)

    masked_node_num=len(masked_index)
    shuffle_criterion=list(range(masked_node_num))
    random.shuffle(shuffle_criterion)

    train_val_test_list=[int(i) for i in ratio.split('-')]
    tvt_sum=sum(train_val_test_list)
    tvt_ratio_list=[i/tvt_sum for i in train_val_test_list]
    #TODO:支持对masked_node_num数不足的情况下进行处理
    train_end_index=int(tvt_ratio_list[0]*masked_node_num)
    val_end_index=train_end_index+int(tvt_ratio_list[1]*masked_node_num)

    train_mask_index=shuffle_criterion[:train_end_index]
    val_mask_index=shuffle_criterion[train_end_index:val_end_index]
    test_mask_index=shuffle_criterion[val_end_index:]

    train_mask=torch.zeros(total_node_num,dtype=torch.bool)
    train_mask[masked_index[train_mask_index]]=True
    val_mask=torch.zeros(total_node_num,dtype=torch.bool)
    val_mask[masked_index[val_mask_index]]=True
    test_mask=torch.zeros(total_node_num,dtype=torch.bool)
    test_mask[masked_index[test_mask_index]]=True

    return (train_mask,val_mask,test_mask)


