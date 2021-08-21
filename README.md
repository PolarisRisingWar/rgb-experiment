本项目/文件夹只包含代码和配置文件等。数据文件未包含。

[TOC]
（注：我用VSCode的MPE看的_(:з」∠)_别的Markdown预览方式可能不支持部分语法）

# 1. quick start
## 1.1 引用数据的方式
1. experiment()函数中数据的默认路径可通过initial_datapath.py中 `default_data_path` 参数修改（**建议使用的方式**）。
2. 类Planetoid格式，可使用zjutoid.py和iozjutoid.py处理：可直接将文件根目录路径传入 `experiment()`，要求目录root下raw文件夹下装载数据文件，root名称即为数据集名称，数据文件中间名需要是数据集名称的全小写格式
3. `x.pt` &nbsp; `y.pt` &nbsp; `edge_index.pt`（Tensor.save()结果）形式数据可使用zjutoid2.py处理为 `torch_geometric.data.Data` 格式。详情可参考该文件注释。
4. `torch_geometric.data.Data` 格式数据：可直接将数据传入 `experiment()`。
您可以提前在data中添加 `train_mask/val_mask/test_mask` 属性（要求是尺寸为 [num_nodes] 的布尔tensor），也可以通过设置 `remake_data_mask=True` 来命令 `experiment()` 自动配置相应属性。
## 1.2 调用函数的方式
1. 建议直接在本目录下运行代码，引入 `experiment()` 函数。示例代码：`from itexperiments import experiment`
2. 如果您希望在其他地方运行，建议使用 `sys.path.extend()` 方法将该目录引入，然后您就可以直接使用上一步的示例代码引入 `experiment()` 函数了。
3. 可以通过代码直接传入参数（见示例文件 run_example1.py），也可以通过配置文件传入参数（见示例文件 ini_example.py）
## 1.3 增加模型的方式
1. 模型文件都放在了 `.models` 文件夹下，可直接将含有以 `torch.nn.Module` 为基类 的模型类文件放在该文件夹中，并在该文件夹下的`__init__.py` 中引用模型，然后在 `exeriment()` 中引用该模型，并在 `if model_name==` 判断语句中增加对该模型的处理即可。
2. 一般来说，您的模型需要返回一个字典，其中键x的value是输出值，键emb的value建议是输出层前一层的节点嵌入值。
3. 如果您对模型有特殊需求，可以参考PTA模型的情况，来定制化地完成一些工作

# 2. 本项目中各文件的简介
1. README.md 介绍项目情况
2. zjutoid.py 和 iozjutoid.py 源自奕青同学（我对其中划分数据集部分代码做了修改），可用于处理类Planetoid格式的图数据集文件。
3. zjutoid2.py 是我写的，见本文档1.1部分介绍。
3. initial_params.py 用于初始化一些数据集的路径（建议使用以防造成更改数据集位置而需要大量修改代码的问题）
4. 文件夹 ini_files：放置示例配置文件，可使用配置文件向 `experiment()` 函数传入参数。使用配置文件的示例见 ini_example.py
5. 文件夹 pics：默认输出图片放置位置
6. itexperiments.py：内含 `experiment()` 函数及其使用的 `test()` 和 `compare_pred_label()` 函数
7. visualize_feature.py：内含 `experiment()` 函数中调用的 `visualize_feature()` 函数，用于绘制节点特征降维到2维平面上的图像。
8. auto_tune_hp.py：内含可调用的自动调参函数（其实不好用），使用示例见 try_auto_tune_hp.py
9. run_example1.py：一个典型的对 initial_params.py 中加载的所有数据集在所有模型上运行 `experiment()` 得到baseline结果的示例代码文件（输出历史见run_example1_output.out）
11. run_example2.py：类似
10. 最终结果.csv：如题，是各数据集上的baseline accuracy指标结果。OOM就是对应数据集在对应模型上OOM了。（注意1：对有向图数据，额外运行了将该数据转换为无向图后得到的结果）（注意2：不实时更新，需要最新版的结果建议直接联系作者）

# 3. 其他注意事项
1. 本项目中matplotlib绘图默认使用本项目下的黑体文件（SimHei.ttf），在代码里写死了。
2. 各模型必须传入 `experiment()` 的参数 `model_init_param` 键及格式见各模型文件。

# 4. 致谢及其他
1. 感谢黄学长和同组的小伙伴提供陪伴和支持！
2. 本项目的requirements将在随后给出。
3. 部分模型的参数等配置参考了原始论文配套代码或torch_geometric项目提供的例子及集成函数。