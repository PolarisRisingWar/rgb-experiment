import sys

sys.path.extend(['whj_code2/integration_experiment'])

from rgb_experiment import zjutoid

dataset_name='Github'
dataset_root='/data/wanghuijuan/dataset1/zjutoid_ds'
split_ratio='6-2-2'
dataset=zjutoid(root=dataset_root,name=dataset_name,split='ratio',ratio=split_ratio)
data=dataset.data
print(data)
#输出示例：Data(edge_index=[2, 289003], test_mask=[37700], train_mask=[37700], val_mask=[37700], x=[37700, 4005], y=[37700])